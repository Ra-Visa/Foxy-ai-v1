# -*- coding: utf-8 -*-
"""
Khmer Video Dubbing Bot (Tool សម្រាយរឿង)
-----------------------------------------
Pipeline:  Video -> Whisper(tiny) STT -> translators (->km) -> edge-tts (Khmer voice)
           -> moviepy (replace audio) -> auto split (<=115s) -> send via Telegram

Env vars:
    BOT_TOKEN   (required)  Telegram bot token from @BotFather
    PORT        (optional)  Flask keep-alive port, default 10000
"""

import os
import sys
import gc
import json
import glob
import time
import wave
import queue
import shutil
import asyncio
import logging
import tempfile
import threading
import subprocess
from datetime import datetime, timedelta

import numpy as np
import telebot
from telebot import types
from telebot.apihelper import ApiTelegramException
from flask import Flask

import whisper
import translators as ts
import edge_tts
import imageio_ffmpeg
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
ADMIN_ID = 6936728139
PORT = int(os.environ.get("PORT", 10000))

WHISPER_MODEL = "tiny"          # តូចបំផុត សន្សំ RAM
MAX_PART_SECONDS = 115          # កាត់ជាភាគ ≤ 115 វិនាទី
MAX_DOWNLOAD_MB = 20            # កំណត់ Telegram Bot API (getFile)
MAX_SPEEDUP = 1.45              # ល្បឿនអតិបរមា នៅពេលសំឡេងខ្មែរវែងជាងពេលវេលា
TRANSLATORS = ["google", "bing", "alibaba"]   # សាកល្បងតាមលំដាប់

VOICES = {
    "male": ("km-KH-PisethNeural", "👨 សំឡេងប្រុស"),
    "female": ("km-KH-SreymomNeural", "👩 សំឡេងស្រី"),
}
LANGS = {
    "auto": "🌐 ស្វ័យប្រវត្តិ",
    "zh": "🇨🇳 ចិន",
    "en": "🇬🇧 អង់គ្លេស",
}

DB_FILE = "users.json"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("dubbing-bot")
logging.getLogger("werkzeug").setLevel(logging.WARNING)

if not BOT_TOKEN:
    print("❌ សូមកំណត់ Environment Variable: BOT_TOKEN")
    sys.exit(1)

bot = telebot.TeleBot(BOT_TOKEN, threaded=True, num_threads=8)


# ----------------------------------------------------------------------------
# USER DATABASE (JSON)
# ----------------------------------------------------------------------------
db_lock = threading.Lock()


def now_kh():
    return datetime.utcnow() + timedelta(hours=7)


def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            log.exception("មិនអាចអាន users.json")
    return {}


users = load_db()


def save_db():
    with db_lock:
        try:
            tmp = DB_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(users, f, ensure_ascii=False, indent=1)
            os.replace(tmp, DB_FILE)
        except Exception:
            log.exception("មិនអាចរក្សាទុក users.json")


def register_user(u):
    uid = str(u.id)
    today = now_kh().strftime("%Y-%m-%d")
    if uid not in users:
        users[uid] = {
            "name": (u.first_name or "") + (" " + u.last_name if u.last_name else ""),
            "username": u.username or "",
            "joined": now_kh().strftime("%Y-%m-%d %H:%M"),
            "last_seen": today,
            "videos": 0,
            "voice": "female",
            "lang": "auto",
        }
        save_db()
    else:
        users[uid]["last_seen"] = today
    return users[uid]


# ----------------------------------------------------------------------------
# KEEP-ALIVE WEB SERVER (Render)
# ----------------------------------------------------------------------------
def keep_alive_server():
    app = Flask(__name__)

    @app.route("/")
    def home():
        return "Khmer Dubbing Bot is running ✅", 200

    @app.route("/health")
    def health():
        return "ok", 200

    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def safe_edit(chat_id, message_id, text):
    try:
        bot.edit_message_text(text, chat_id, message_id)
    except Exception:
        pass


def settings_markup(uid):
    p = users[str(uid)]
    kb = types.InlineKeyboardMarkup()
    kb.row(*[
        types.InlineKeyboardButton(
            ("✅ " if p["voice"] == k else "") + v[1], callback_data=f"voice:{k}"
        )
        for k, v in VOICES.items()
    ])
    kb.row(*[
        types.InlineKeyboardButton(
            ("✅ " if p["lang"] == k else "") + v, callback_data=f"lang:{k}"
        )
        for k, v in LANGS.items()
    ])
    return kb


SETTINGS_TEXT = (
    "⚙️ ការកំណត់\n\n"
    "• ជ្រើសរើសសំឡេង AI ខ្មែរ (ប្រុស/ស្រី)\n"
    "• ជ្រើសរើសភាសាដើមក្នុងវីដេអូ (បើ 'ស្វ័យប្រវត្តិ' មិនត្រឹមត្រូវ សូមជ្រើសភាសាច្បាស់លាស់)"
)


# ----------------------------------------------------------------------------
# AI PIPELINE
# ----------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            log.info("កំពុងផ្ទុក Whisper model: %s", WHISPER_MODEL)
            _model = whisper.load_model(WHISPER_MODEL, device="cpu")
        return _model


def load_wav(path):
    with wave.open(path, "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(wav_path, lang):
    audio = load_wav(wav_path)
    kwargs = dict(
        task="transcribe",
        fp16=False,
        verbose=None,
        condition_on_previous_text=False,
    )
    if lang in ("zh", "en"):
        kwargs["language"] = lang
    result = get_model().transcribe(audio, **kwargs)
    segs = []
    for s in result.get("segments", []):
        text = s["text"].strip()
        if text and any(ch.isalnum() for ch in text):
            segs.append({"start": float(s["start"]), "end": float(s["end"]), "text": text})
    return result.get("language", lang), segs


def translate_km(text):
    for provider in TRANSLATORS:
        try:
            out = ts.translate_text(
                text,
                translator=provider,
                from_language="auto",
                to_language="km",
            )
            if out and str(out).strip():
                return str(out).strip()
        except Exception as e:
            log.warning("translator %s បរាជ័យ: %s", provider, str(e)[:120])
    return None


async def _synth_all(texts, voice, folder):
    paths = []
    for i, text in enumerate(texts):
        path = os.path.join(folder, f"tts_{i}.mp3")
        ok = False
        for _ in range(3):
            try:
                await edge_tts.Communicate(text, voice).save(path)
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    ok = True
                    break
            except Exception as e:
                log.warning("edge-tts ម្ដងទៀត: %s", str(e)[:120])
                await asyncio.sleep(1.5)
        paths.append(path if ok else None)
    return paths


def synthesize(texts, voice, folder):
    return asyncio.run(_synth_all(texts, voice, folder))


def speed_up(src, dst, factor):
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", src,
         "-filter:a", f"atempo={factor:.3f}", dst],
        check=True,
    )


def build_audio_track(segs, paths, total_dur, folder):
    """ដាក់សំឡេងខ្មែរតាមពេលវេលាដើម មិនឱ្យជាន់គ្នា"""
    clips = []
    cursor = 0.0
    for idx, (seg, p) in enumerate(zip(segs, paths)):
        if not p:
            continue
        try:
            clip = AudioFileClip(p)
        except Exception:
            continue
        dur = clip.duration
        start = max(seg["start"], cursor)
        if start >= total_dur:
            clip.close()
            break
        next_start = segs[idx + 1]["start"] if idx + 1 < len(segs) else total_dur
        slot = max(next_start - start, 0.6)
        if dur > slot:
            factor = min(dur / slot, MAX_SPEEDUP)
            if factor > 1.05:
                try:
                    clip.close()
                    fast = os.path.join(folder, f"tts_{idx}_fast.mp3")
                    speed_up(p, fast, factor)
                    clip = AudioFileClip(fast)
                    dur = clip.duration
                except Exception:
                    clip = AudioFileClip(p)
                    dur = clip.duration
        clips.append(clip.set_start(start))
        cursor = start + dur + 0.05
    if not clips:
        return None, []
    track = CompositeAudioClip(clips).set_duration(total_dur)
    track.fps = 44100
    return track, clips


def split_video(path, folder):
    pattern = os.path.join(folder, "part_%03d.mp4")
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", path,
         "-map", "0", "-c", "copy", "-f", "segment",
         "-segment_time", str(MAX_PART_SECONDS),
         "-reset_timestamps", "1", pattern],
        check=True,
    )
    return sorted(glob.glob(os.path.join(folder, "part_*.mp4")))


def send_file(chat_id, path, caption):
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > 49:
        bot.send_message(chat_id, f"⚠️ {caption}: ឯកសារធំពេក ({size_mb:.0f}MB) Telegram មិនអនុញ្ញាតឱ្យផ្ញើ។")
        return
    for attempt in range(3):
        try:
            with open(path, "rb") as f:
                bot.send_video(chat_id, f, caption=caption,
                               supports_streaming=True, timeout=600)
            return
        except Exception as e:
            log.warning("send_video បរាជ័យ (%s): %s", attempt + 1, str(e)[:150])
            time.sleep(3)
    try:
        with open(path, "rb") as f:
            bot.send_document(chat_id, f, caption=caption, timeout=600)
    except Exception:
        bot.send_message(chat_id, f"❌ មិនអាចផ្ញើ {caption} បានទេ។")


# ----------------------------------------------------------------------------
# JOB QUEUE (ដំណើរការម្ដងមួយវីដេអូ ដើម្បីសន្សំ RAM)
# ----------------------------------------------------------------------------
jobs = queue.Queue()


def process_job(job):
    chat_id = job["chat_id"]
    status_id = job["status_id"]
    user_id = job["user_id"]

    def st(text):
        safe_edit(chat_id, status_id, text)

    prefs = users.get(str(user_id), {})
    voice = VOICES.get(prefs.get("voice", "female"), VOICES["female"])[0]
    lang = prefs.get("lang", "auto")

    tmp = tempfile.mkdtemp(prefix="dub_")
    video = None
    track = None
    tts_clips = []
    final_v = None
    try:
        # 1) Download
        st("📥 [1/6] កំពុងទាញយកវីដេអូ...")
        info = bot.get_file(job["file_id"])
        data = bot.download_file(info.file_path)
        src = os.path.join(tmp, "input.mp4")
        with open(src, "wb") as f:
            f.write(data)
        del data

        video = VideoFileClip(src)
        total = float(video.duration)
        if video.audio is None:
            st("❌ វីដេអូនេះគ្មានសំឡេងទេ។")
            return

        # 2) Extract audio
        st("🎧 [2/6] កំពុងស្រង់សំឡេង...")
        wav = os.path.join(tmp, "audio.wav")
        video.audio.write_audiofile(
            wav, fps=16000, nbytes=2, codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"], logger=None,
        )

        # 3) STT
        st("📝 [3/6] កំពុងស្រង់អត្ថបទ (Whisper)... អាចចំណាយពេលបន្តិច")
        detected, segs = transcribe(wav, lang)
        try:
            os.remove(wav)
        except Exception:
            pass
        if not segs:
            st("❌ រកមិនឃើញសំឡេងនិយាយក្នុងវីដេអូនេះទេ។")
            return

        # 4) Translate
        st(f"🌐 [4/6] កំពុងបកប្រែជាភាសាខ្មែរ ({len(segs)} ឃ្លា, ភាសាដើម: {detected})...")
        cache = {}
        kept_segs, kh_texts = [], []
        for s in segs:
            t = s["text"]
            if t not in cache:
                cache[t] = translate_km(t)
            if cache[t]:
                kept_segs.append(s)
                kh_texts.append(cache[t])
        if not kh_texts:
            st("❌ ការបកប្រែបរាជ័យ។ សូមព្យាយាមម្ដងទៀតពេលក្រោយ។")
            return

        # 5) TTS
        st("🎙 [5/6] កំពុងបង្កើតសំឡេង AI ខ្មែរ...")
        paths = synthesize(kh_texts, voice, tmp)
        track, tts_clips = build_audio_track(kept_segs, paths, total, tmp)
        if track is None:
            st("❌ បង្កើតសំឡេងខ្មែរមិនបានទេ។ សូមព្យាយាមម្ដងទៀត។")
            return

        # 6) Render
        st("🎬 [6/6] កំពុង Render វីដេអូ... សូមរង់ចាំ")
        out = os.path.join(tmp, "dubbed.mp4")
        params = ["-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if video.h and video.h > 720:
            params += ["-vf", "scale=-2:720"]
        if total > MAX_PART_SECONDS:
            params += ["-force_key_frames", f"expr:gte(t,n_forced*{MAX_PART_SECONDS})"]
        final_v = video.without_audio().set_audio(track)
        final_v.write_videofile(
            out,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=2,
            fps=min(video.fps or 24, 30),
            ffmpeg_params=params,
            temp_audiofile=os.path.join(tmp, "tmp_audio.m4a"),
            logger=None,
        )

        # Split + send
        if total > MAX_PART_SECONDS:
            st("✂️ វីដេអូវែង កំពុងកាត់ជាភាគតូចៗ...")
            parts = split_video(out, tmp)
        else:
            parts = [out]

        st(f"📤 កំពុងផ្ញើ {len(parts)} ភាគ...")
        for i, part in enumerate(parts, 1):
            cap = "✅ សម្រាយជាភាសាខ្មែរ" if len(parts) == 1 else f"✅ ភាគទី {i}/{len(parts)}"
            send_file(chat_id, part, cap)

        st("✅ រួចរាល់! សូមអរគុណដែលប្រើប្រាស់ Bot។")
        if str(user_id) in users:
            users[str(user_id)]["videos"] = users[str(user_id)].get("videos", 0) + 1
            save_db()

    except Exception as e:
        log.exception("process_job error")
        st(f"❌ មានបញ្ហាក្នុងការដំណើរការ: {str(e)[:200]}")
    finally:
        for obj in (final_v, track, video):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        for c in tts_clips:
            try:
                c.close()
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)
        gc.collect()


def worker():
    try:
        get_model()  # preload
    except Exception:
        log.exception("preload Whisper បរាជ័យ")
    while True:
        job = jobs.get()
        try:
            process_job(job)
        except Exception:
            log.exception("worker error")
        finally:
            jobs.task_done()


# ----------------------------------------------------------------------------
# USER COMMANDS
# ----------------------------------------------------------------------------
@bot.message_handler(commands=["start"])
def cmd_start(m):
    register_user(m.from_user)
    bot.send_message(
        m.chat.id,
        f"សួស្តី {m.from_user.first_name}! 👋\n\n"
        "ខ្ញុំជា Bot សម្រាយរឿង (Dubbing) ស្វ័យប្រវត្តិ។\n\n"
        "🎬 ផ្ញើវីដេអូដែលមានសំឡេងភាសាចិន ឬអង់គ្លេសមកខ្ញុំ "
        "ហើយខ្ញុំនឹងបំប្លែងជាសំឡេងនិយាយភាសាខ្មែរ។\n\n"
        "⚙️ /settings — ជ្រើសរើសសំឡេង (ប្រុស/ស្រី) និងភាសាដើម\n"
        "❓ /help — ជំនួយ\n\n"
        f"📌 វីដេអូមិនត្រូវលើស {MAX_DOWNLOAD_MB}MB ទេ។ "
        "បើវីដេអូវែងជាង ២ នាទី ខ្ញុំនឹងកាត់ជាភាគតូចៗឱ្យស្វ័យប្រវត្តិ។",
    )


@bot.message_handler(commands=["help"])
def cmd_help(m):
    register_user(m.from_user)
    bot.send_message(
        m.chat.id,
        "📖 របៀបប្រើ៖\n"
        "1. ផ្ញើវីដេអូ (ចិន ឬ អង់គ្លេស)\n"
        "2. រង់ចាំ Bot ដំណើរការ (ស្រង់អត្ថបទ → បកប្រែ → សំឡេង AI → Render)\n"
        "3. ទទួលវីដេអូសម្រាយជាភាសាខ្មែរ\n\n"
        "⚙️ /settings ដើម្បីប្តូរសំឡេង/ភាសា\n"
        "⏳ Bot ដំណើរការម្ដងមួយវីដេអូ បើមានអ្នកផ្សេងកំពុងប្រើ សូមរង់ចាំក្នុងជួរ។",
    )


@bot.message_handler(commands=["settings"])
def cmd_settings(m):
    register_user(m.from_user)
    bot.send_message(m.chat.id, SETTINGS_TEXT, reply_markup=settings_markup(m.from_user.id))


# ----------------------------------------------------------------------------
# ADMIN PANEL & BROADCAST
# ----------------------------------------------------------------------------
def is_admin(uid):
    return uid == ADMIN_ID


def stats_text():
    today = now_kh().strftime("%Y-%m-%d")
    total = len(users)
    active = sum(1 for u in users.values() if u.get("last_seen") == today)
    new_today = sum(1 for u in users.values() if u.get("joined", "").startswith(today))
    videos = sum(u.get("videos", 0) for u in users.values())
    return (
        "📊 ស្ថិតិ Bot\n\n"
        f"👥 អ្នកប្រើសរុប: {total}\n"
        f"🟢 សកម្មថ្ងៃនេះ: {active}\n"
        f"🆕 ចូលថ្មីថ្ងៃនេះ: {new_today}\n"
        f"🎬 វីដេអូដែលបានដំណើរការ: {videos}\n"
        f"⏳ ក្នុងជួររង់ចាំ: {jobs.qsize()}"
    )


def admin_markup():
    kb = types.InlineKeyboardMarkup()
    kb.row(
        types.InlineKeyboardButton("📊 ស្ថិតិ", callback_data="admin:stats"),
        types.InlineKeyboardButton("📢 Broadcast", callback_data="admin:broadcast"),
    )
    kb.row(types.InlineKeyboardButton("👥 អ្នកប្រើចុងក្រោយ", callback_data="admin:users"))
    return kb


@bot.message_handler(commands=["admin"])
def cmd_admin(m):
    if not is_admin(m.from_user.id):
        return
    bot.send_message(m.chat.id, "🛠 Admin Panel", reply_markup=admin_markup())


@bot.message_handler(commands=["stats"])
def cmd_stats(m):
    if is_admin(m.from_user.id):
        bot.send_message(m.chat.id, stats_text())


@bot.message_handler(commands=["broadcast"])
def cmd_broadcast(m):
    if is_admin(m.from_user.id):
        ask_broadcast(m.chat.id)


def ask_broadcast(chat_id):
    msg = bot.send_message(
        chat_id,
        "📢 សូមផ្ញើសារដែលចង់ Broadcast (អត្ថបទ/រូប/វីដេអូ ក៏បាន)។\n"
        "វាយ /cancel ដើម្បីបោះបង់។",
    )
    bot.register_next_step_handler(msg, do_broadcast)


def do_broadcast(m):
    if not is_admin(m.from_user.id):
        return
    if m.text and m.text.strip().lower().startswith("/cancel"):
        bot.send_message(m.chat.id, "🚫 បានបោះបង់ Broadcast។")
        return
    threading.Thread(target=broadcast_worker, args=(m,), daemon=True).start()


def broadcast_worker(m):
    ids = list(users.keys())
    bot.send_message(m.chat.id, f"📡 កំពុងផ្ញើទៅ {len(ids)} នាក់...")
    ok = fail = 0
    for uid in ids:
        try:
            bot.copy_message(int(uid), m.chat.id, m.message_id)
            ok += 1
        except ApiTelegramException as e:
            fail += 1
            log.info("broadcast fail %s: %s", uid, str(e)[:80])
        except Exception:
            fail += 1
        time.sleep(0.06)  # ជៀសវាង flood limit
    bot.send_message(m.chat.id, f"✅ Broadcast រួចរាល់\n📨 ជោគជ័យ: {ok}\n⚠️ បរាជ័យ: {fail}")


# ----------------------------------------------------------------------------
# CALLBACKS
# ----------------------------------------------------------------------------
@bot.callback_query_handler(func=lambda c: True)
def on_callback(c):
    try:
        data = c.data or ""
        uid = c.from_user.id

        if data.startswith("admin:"):
            if not is_admin(uid):
                bot.answer_callback_query(c.id, "⛔ សម្រាប់ Admin ប៉ុណ្ណោះ", show_alert=True)
                return
            action = data.split(":", 1)[1]
            bot.answer_callback_query(c.id)
            if action == "stats":
                bot.send_message(c.message.chat.id, stats_text())
            elif action == "broadcast":
                ask_broadcast(c.message.chat.id)
            elif action == "users":
                last = list(users.items())[-10:]
                lines = ["👥 អ្នកប្រើ ១០ នាក់ចុងក្រោយ\n"]
                for k, u in reversed(last):
                    uname = f"@{u['username']}" if u.get("username") else "-"
                    lines.append(f"• {u.get('name','')} {uname} | ID: {k} | 🎬 {u.get('videos',0)}")
                bot.send_message(c.message.chat.id, "\n".join(lines))
            return

        if data.startswith("voice:") or data.startswith("lang:"):
            register_user(c.from_user)
            kind, val = data.split(":", 1)
            if kind == "voice" and val in VOICES:
                users[str(uid)]["voice"] = val
            elif kind == "lang" and val in LANGS:
                users[str(uid)]["lang"] = val
            else:
                bot.answer_callback_query(c.id)
                return
            save_db()
            try:
                bot.edit_message_reply_markup(
                    c.message.chat.id, c.message.message_id,
                    reply_markup=settings_markup(uid),
                )
            except Exception:
                pass
            bot.answer_callback_query(c.id, "✅ បានរក្សាទុក")
            return

        bot.answer_callback_query(c.id)
    except Exception:
        log.exception("callback error")


# ----------------------------------------------------------------------------
# VIDEO HANDLER
# ----------------------------------------------------------------------------
@bot.message_handler(content_types=["video", "video_note", "document"])
def on_video(m):
    register_user(m.from_user)

    if m.content_type == "video":
        f = m.video
    elif m.content_type == "video_note":
        f = m.video_note
    else:
        f = m.document
        if not (f.mime_type or "").startswith("video/"):
            bot.reply_to(m, "⚠️ សូមផ្ញើជាឯកសារវីដេអូ (mp4, mov, mkv ...)។")
            return

    if f.file_size and f.file_size > MAX_DOWNLOAD_MB * 1024 * 1024:
        bot.reply_to(
            m,
            f"⚠️ វីដេអូធំពេក ({f.file_size / 1024 / 1024:.1f}MB)។ "
            f"Bot ទទួលបានត្រឹម {MAX_DOWNLOAD_MB}MB ប៉ុណ្ណោះ។ សូមបង្រួមវីដេអូ ឬកាត់ជាភាគតូចៗ។",
        )
        return

    pos = jobs.qsize() + 1
    status = bot.reply_to(
        m,
        "⏳ បានទទួលវីដេអូ! "
        + ("កំពុងចាប់ផ្ដើម..." if pos == 1 else f"អ្នកនៅលំដាប់ទី {pos} ក្នុងជួររង់ចាំ។"),
    )
    jobs.put({
        "chat_id": m.chat.id,
        "status_id": status.message_id,
        "user_id": m.from_user.id,
        "file_id": f.file_id,
    })


@bot.message_handler(content_types=["text"])
def on_text(m):
    register_user(m.from_user)
    bot.reply_to(m, "🎬 សូមផ្ញើវីដេអូមកខ្ញុំ ដើម្បីសម្រាយជាភាសាខ្មែរ។ ប្រើ /help សម្រាប់ជំនួយ។")


# ----------------------------------------------------------------------------
# RUN 24/7
# ----------------------------------------------------------------------------
def run_forever():
    while True:
        try:
            bot.delete_webhook(drop_pending_updates=True)
            log.info("🤖 Bot កំពុងដំណើរការ...")
            bot.infinity_polling(
                timeout=30,
                long_polling_timeout=30,
                skip_pending=True,
                allowed_updates=["message", "callback_query"],
            )
        except Exception as e:
            log.exception("polling error")
            if "409" in str(e):
                log.warning("Error 409 Conflict — រង់ចាំ 15 វិនាទី រួចសាកម្ដងទៀត")
                time.sleep(15)
            else:
                time.sleep(5)


if __name__ == "__main__":
    threading.Thread(target=keep_alive_server, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    run_forever()
