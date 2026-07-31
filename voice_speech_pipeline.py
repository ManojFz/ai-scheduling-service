"""
Azure Speech (STT/TTS) + GPT-4.1 Nano voice pipeline for Exotel calls.
Replaces Azure OpenAI Realtime when VOICE_BACKEND=speech.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:  # pragma: no cover
    speechsdk = None  # type: ignore

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Populated by service.py after its globals exist (avoids circular imports at load time).
_svc: Any = None

speech_sessions: Dict[str, Dict[str, Any]] = {}

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")
AZURE_SPEECH_ENDPOINT = os.getenv("AZURE_SPEECH_ENDPOINT", "").rstrip("/")
AZURE_SPEECH_STT_LOCALES = [
    x.strip()
    for x in os.getenv("AZURE_SPEECH_STT_LOCALES", "en-IN").split(",")
    if x.strip()
]
AZURE_SPEECH_VOICE = os.getenv("AZURE_SPEECH_VOICE", "en-IN-NeerjaNeural")
AZURE_SPEECH_VOICE_HI = os.getenv("AZURE_SPEECH_VOICE_HI", "hi-IN-SwaraNeural")
AZURE_SPEECH_VOICE_KN = os.getenv("AZURE_SPEECH_VOICE_KN", "kn-IN-SapnaNeural")

PCM_8K_BYTES_PER_SECOND = 16000  # 8 kHz * 2 bytes

# Silence nudge / hangup timings (seconds) — mirrored in service.py
SILENCE_NUDGE_SECONDS = float(os.getenv("SILENCE_NUDGE_SECONDS", "5"))
SILENCE_HANGUP_SECONDS = float(os.getenv("SILENCE_HANGUP_SECONDS", "25"))
LISTEN_GRACE_AFTER_PLAYBACK_SECONDS = float(
    os.getenv("LISTEN_GRACE_AFTER_PLAYBACK_SECONDS", "0.4")
)


def _reload_speech_env() -> None:
    """Refresh module-level Speech settings (after dotenv / runtime env changes)."""
    global AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, AZURE_SPEECH_ENDPOINT
    global AZURE_SPEECH_STT_LOCALES, AZURE_SPEECH_VOICE, AZURE_SPEECH_VOICE_HI, AZURE_SPEECH_VOICE_KN
    AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
    AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")
    AZURE_SPEECH_ENDPOINT = os.getenv("AZURE_SPEECH_ENDPOINT", "").rstrip("/")
    AZURE_SPEECH_STT_LOCALES = [
        x.strip()
        for x in os.getenv("AZURE_SPEECH_STT_LOCALES", "en-IN").split(",")
        if x.strip()
    ]
    AZURE_SPEECH_VOICE = os.getenv("AZURE_SPEECH_VOICE", "en-IN-NeerjaNeural")
    AZURE_SPEECH_VOICE_HI = os.getenv("AZURE_SPEECH_VOICE_HI", "hi-IN-SwaraNeural")
    AZURE_SPEECH_VOICE_KN = os.getenv("AZURE_SPEECH_VOICE_KN", "kn-IN-SapnaNeural")


GREETING_USER_PROMPT = (
    "(The phone call has just connected. Begin immediately with Step 0a, then Step 0b "
    "from the script. Do not wait for the customer to speak first. Do not skip any opening step.)"
)

SILENCE_USER_PROMPT = (
    "(The customer has been silent for a while after you finished speaking. "
    "Say 'Are you still there?' and then repeat your last question exactly.)"
)


def _cancel_silence_watchdog(stream_sid: str) -> None:
    task = _svc.silence_timer_tasks.pop(stream_sid, None)
    if task and not task.done():
        # Never cancel the currently running watchdog from inside itself —
        # that aborts the nudge TTS and prevents the 15s hangup.
        try:
            if task is asyncio.current_task():
                return
        except Exception:
            pass
        task.cancel()


def _session_alive_for_silence(stream_sid: str, gen: int) -> bool:
    """True if this silence-watchdog generation should keep running."""
    sess = speech_sessions.get(stream_sid)
    if not sess or sess.get("closed") or int(sess.get("silence_gen", 0)) != gen:
        return False
    ctx = _svc.call_context.get(stream_sid) or {}
    if ctx.get("status") == "closing" or ctx.get("pending_hangup"):
        return False
    return True


def _watchdog_still_valid(stream_sid: str, gen: int) -> bool:
    """Valid for firing nudge/hangup: session alive and currently in listen window."""
    if not _session_alive_for_silence(stream_sid, gen):
        return False
    sess = speech_sessions.get(stream_sid)
    if not sess:
        return False
    if sess.get("bot_speaking") or not sess.get("listening_enabled"):
        return False
    return True


def _user_spoke_since_listen_start(stream_sid: str) -> bool:
    sess = speech_sessions.get(stream_sid)
    if not sess:
        return False
    listen_start = float(sess.get("listen_started_at") or 0)
    last_speech = float(sess.get("last_user_speech_at") or 0)
    return last_speech > listen_start + 0.05


def _user_active_since(stream_sid: str, ts: float) -> bool:
    """True if any user speech was recognized after `ts` — including speech that
    arrived while the listen gate was closed (dropped by half-duplex). A customer
    who spoke during bot playback is present and must not be hung up on."""
    sess = speech_sessions.get(stream_sid)
    if not sess:
        return False
    last = max(
        float(sess.get("last_user_speech_at") or 0),
        float(sess.get("last_user_activity_at") or 0),
    )
    return last > ts + 0.05


def _mark_user_activity(stream_sid: str) -> None:
    sess = speech_sessions.get(stream_sid)
    if sess:
        sess["last_user_activity_at"] = time.time()


async def _wait_tts_gen_playback(stream_sid: str, gen: int) -> None:
    """Block until TTS `gen` has finished playing (or session ended)."""
    deadline = time.time() + 60.0
    while time.time() < deadline:
        sess = speech_sessions.get(stream_sid)
        if not sess or sess.get("closed") or sess.get("tts_gen") != gen:
            return
        ev = (sess.get("playback_done_events") or {}).get(gen)
        if ev and ev.is_set():
            return
        tracking = _svc.response_audio_tracking.get(stream_sid) or {}
        ends_at = tracking.get("playback_ends_at")
        if ends_at and time.time() >= float(ends_at) + LISTEN_GRACE_AFTER_PLAYBACK_SECONDS:
            return
        await asyncio.sleep(0.05)


def _enable_listening(stream_sid: str) -> None:
    sess = speech_sessions.get(stream_sid)
    if not sess or sess.get("closed"):
        return
    sess["bot_speaking"] = False
    sess["listening_enabled"] = True
    sess["listen_started_at"] = time.time()


def _disable_listening(stream_sid: str) -> None:
    sess = speech_sessions.get(stream_sid)
    if not sess:
        return
    sess["listening_enabled"] = False


def schedule_listening_silence_watchdog(stream_sid: str) -> None:
    """Start 5s nudge / 15s hangup timers after bot finishes speaking."""
    sess = speech_sessions.get(stream_sid)
    if not sess or sess.get("closed"):
        return
    ctx = _svc.call_context.get(stream_sid) or {}
    if ctx.get("status") == "closing" or ctx.get("pending_hangup"):
        return
    gen = int(sess.get("silence_gen", 0)) + 1
    sess["silence_gen"] = gen
    sess["listen_started_at"] = time.time()
    sess["silence_nudged"] = False
    _cancel_silence_watchdog(stream_sid)
    _svc.silence_timer_tasks[stream_sid] = asyncio.create_task(
        _listening_silence_watchdog(stream_sid, gen)
    )
    logger.info(
        "Silence watchdog started stream=%s gen=%s (nudge=%ss hangup=%ss)",
        stream_sid,
        gen,
        SILENCE_NUDGE_SECONDS,
        SILENCE_HANGUP_SECONDS,
    )


async def _speak_direct(
    stream_sid: str,
    text: str,
    *,
    schedule_silence_after: bool,
    cancel_silence: bool = False,
) -> None:
    """TTS-only line (no Nano). Used for silence nudge / silence hangup."""
    if not text:
        return
    await _synthesize_and_queue(
        stream_sid,
        text,
        schedule_silence_after=schedule_silence_after,
        cancel_silence=cancel_silence,
    )
    sess = speech_sessions.get(stream_sid)
    if not sess:
        return
    gen = int(sess.get("tts_gen", 0))
    await _wait_tts_gen_playback(stream_sid, gen)
    # Ensure listen window is open after a silence-owned speak
    if not schedule_silence_after:
        _enable_listening(stream_sid)


async def _speak_still_there_nudge(stream_sid: str) -> None:
    text = _svc.build_still_there_repeat(stream_sid)
    logger.info("Silence nudge stream=%s: %r", stream_sid, text)
    # Do NOT cancel the running watchdog — we are inside it.
    await _speak_direct(
        stream_sid,
        text,
        schedule_silence_after=False,
        cancel_silence=False,
    )


async def _silence_hangup(stream_sid: str) -> None:
    if stream_sid not in speech_sessions or stream_sid not in _svc.call_context:
        return
    ctx = _svc.call_context[stream_sid]
    if ctx.get("status") == "closing" or ctx.get("pending_hangup"):
        return
    ctx["pending_hangup"] = True
    text = _svc.build_silence_hangup_message(stream_sid)
    logger.warning(
        "Silence hangup stream=%s after %ss: %r",
        stream_sid,
        SILENCE_HANGUP_SECONDS,
        text,
    )
    ctx["comments"] = "No response; call disconnected due to silence."
    ctx["slotSelected"] = False
    await _speak_direct(
        stream_sid,
        text,
        schedule_silence_after=False,
        cancel_silence=False,
    )
    # Run cleanup outside this watchdog task so self-cancel cannot abort hangup.
    asyncio.create_task(_svc.cleanup_after_playback(stream_sid))


def _reschedule_after_activity(stream_sid: str, gen: int) -> None:
    """Re-arm the silence watchdog when we skipped a hangup because of user
    activity that was dropped by the listen gate (no bot turn was started for
    it). Without this the call would sit in dead air with no watchdog. If a
    real turn is in flight (bot speaking / listening closed), its playback
    completion will schedule a fresh watchdog, so do nothing here."""
    if not _session_alive_for_silence(stream_sid, gen):
        return
    sess = speech_sessions.get(stream_sid)
    if not sess or sess.get("bot_speaking") or not sess.get("listening_enabled"):
        return
    schedule_listening_silence_watchdog(stream_sid)


async def _listening_silence_watchdog(stream_sid: str, gen: int) -> None:
    listen_start = time.time()
    try:
        await asyncio.sleep(SILENCE_NUDGE_SECONDS)
        if not _session_alive_for_silence(stream_sid, gen):
            return
        if _user_spoke_since_listen_start(stream_sid) or _user_active_since(stream_sid, listen_start):
            logger.info("Silence nudge skipped — user spoke stream=%s", stream_sid)
            return
        if not _watchdog_still_valid(stream_sid, gen):
            # Brief race with TTS/listen gate; retry shortly if still silent
            await asyncio.sleep(0.5)
            if (
                not _session_alive_for_silence(stream_sid, gen)
                or _user_spoke_since_listen_start(stream_sid)
                or _user_active_since(stream_sid, listen_start)
                or not _watchdog_still_valid(stream_sid, gen)
            ):
                return

        sess = speech_sessions.get(stream_sid)
        if sess:
            sess["silence_nudged"] = True
        await _speak_still_there_nudge(stream_sid)

        if not _session_alive_for_silence(stream_sid, gen):
            return
        if _user_spoke_since_listen_start(stream_sid) or _user_active_since(stream_sid, listen_start):
            logger.info("Silence hangup skipped — user spoke after nudge stream=%s", stream_sid)
            _reschedule_after_activity(stream_sid, gen)
            return

        # Give the customer a FRESH answer window measured from when the nudge
        # finished playing — never "whatever is left" of the original window,
        # because the nudge itself (repeating a long date list) eats that time.
        answer_window = max(5.0, SILENCE_HANGUP_SECONDS - SILENCE_NUDGE_SECONDS)
        nudge_done_at = time.time()
        logger.info(
            "Silence hangup wait stream=%s window=%.1fs after nudge",
            stream_sid,
            answer_window,
        )
        while time.time() - nudge_done_at < answer_window:
            await asyncio.sleep(0.25)
            if not _session_alive_for_silence(stream_sid, gen):
                return
            if _user_spoke_since_listen_start(stream_sid) or _user_active_since(
                stream_sid, listen_start
            ):
                logger.info("Silence hangup skipped — user spoke stream=%s", stream_sid)
                _reschedule_after_activity(stream_sid, gen)
                return

        await _silence_hangup(stream_sid)
    except asyncio.CancelledError:
        logger.info("Silence watchdog cancelled stream=%s gen=%s", stream_sid, gen)
    except Exception as e:
        logger.warning("Silence watchdog error for %s: %s", stream_sid, e, exc_info=True)
    finally:
        cur = _svc.silence_timer_tasks.get(stream_sid)
        try:
            if cur is asyncio.current_task():
                _svc.silence_timer_tasks.pop(stream_sid, None)
        except Exception:
            _svc.silence_timer_tasks.pop(stream_sid, None)


def bind_service(module: Any) -> None:
    global _svc
    _svc = module


def _require_speech_sdk() -> None:
    _reload_speech_env()
    if speechsdk is None:
        raise RuntimeError(
            "azure-cognitiveservices-speech is not installed. "
            "Add it to requirements and pip install."
        )
    if not AZURE_SPEECH_KEY:
        raise RuntimeError("AZURE_SPEECH_KEY is not set")


def _speech_config() -> "speechsdk.SpeechConfig":
    _require_speech_sdk()
    if AZURE_SPEECH_ENDPOINT:
        # Host-only endpoint works with region; prefer key+region for SDK stability.
        cfg = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    else:
        cfg = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    cfg.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "15000")
    cfg.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "800")
    cfg.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm
    )
    return cfg


def _pick_tts_voice(stream_sid: Optional[str] = None, text: str = "") -> str:
    """Prefer locked call language so TTS voice does not flip with Nano script mixups."""
    lang = None
    if stream_sid and _svc is not None:
        try:
            lang = _svc.detect_locked_language(stream_sid)
        except Exception:
            lang = None
    if lang == "kn":
        return AZURE_SPEECH_VOICE_KN
    if lang == "hi":
        return AZURE_SPEECH_VOICE_HI
    if lang == "en":
        return AZURE_SPEECH_VOICE
    if re.search(r"[\u0C80-\u0CFF]", text or ""):
        return AZURE_SPEECH_VOICE_KN
    if re.search(r"[\u0900-\u097F]", text or ""):
        return AZURE_SPEECH_VOICE_HI
    return AZURE_SPEECH_VOICE


def split_speakable_text(message: str) -> str:
    """Remove system TAG_* / status tokens so TTS does not speak them."""
    lines = []
    for line in (message or "").splitlines():
        stripped = line.strip()
        if re.fullmatch(r"TAG_[A-Z0-9_]+", stripped, re.I):
            continue
        if re.fullmatch(r"(CONFIRMED|DECLINED|DECLINE|DECLINING)", stripped, re.I):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\bTAG_[A-Z0-9_]+\b", "", text, flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _schedule(loop: asyncio.AbstractEventLoop, coro) -> None:
    try:
        asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception as e:
        logger.error("Failed to schedule coroutine on event loop: %s", e, exc_info=True)


async def start_speech_session(stream_sid: str, system_message: str) -> None:
    _require_speech_sdk()
    if stream_sid in speech_sessions:
        await stop_speech_session(stream_sid)

    loop = asyncio.get_running_loop()
    speech_cfg = _speech_config()

    audio_format = speechsdk.audio.AudioStreamFormat(
        samples_per_second=8000, bits_per_sample=16, channels=1
    )
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    if len(AZURE_SPEECH_STT_LOCALES) > 1:
        auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=AZURE_SPEECH_STT_LOCALES
        )
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_cfg,
            auto_detect_source_language_config=auto_detect,
            audio_config=audio_config,
        )
    else:
        speech_cfg.speech_recognition_language = AZURE_SPEECH_STT_LOCALES[0] if AZURE_SPEECH_STT_LOCALES else "en-IN"
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_cfg, audio_config=audio_config
        )

    session: Dict[str, Any] = {
        "push_stream": push_stream,
        "recognizer": recognizer,
        "history": [SystemMessage(content=system_message)],
        "loop": loop,
        "bot_speaking": False,
        "listening_enabled": False,
        "tts_gen": 0,
        "silence_gen": 0,
        "listen_started_at": 0.0,
        "last_user_speech_at": 0.0,
        "last_user_activity_at": 0.0,
        "silence_nudged": False,
        "playback_done_events": {},
        "turn_lock": asyncio.Lock(),
        "closed": False,
    }
    speech_sessions[stream_sid] = session

    def on_recognizing(evt):
        # Half-duplex: never barge-in while the bot is speaking.
        return

    def on_recognized(evt):
        if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return
        text = (evt.result.text or "").strip()
        if not text:
            return
        sess = speech_sessions.get(stream_sid)
        if not sess or sess.get("closed"):
            return
        if sess.get("bot_speaking") or not sess.get("listening_enabled"):
            # Still counts as presence — never silence-hangup on a customer
            # whose answer arrived while the bot was talking.
            _mark_user_activity(stream_sid)
            logger.info(
                "Ignored STT while bot speaking/not listening: %r for stream %s",
                text,
                stream_sid,
            )
            return
        logger.info("👤 User said (Speech STT): %r for stream %s", text, stream_sid)
        _schedule(loop, _on_user_utterance(stream_sid, text))

    def on_canceled(evt):
        logger.warning(
            "Speech recognition canceled for %s: %s",
            stream_sid,
            evt,
        )

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.start_continuous_recognition_async()

    logger.info("Speech session started for stream %s — triggering greeting", stream_sid)
    # Do not await: keep Exotel WS receive loop free while Nano+TTS run.
    greeting = _svc.greeting_user_prompt_for_stream(stream_sid)
    asyncio.create_task(inject_and_respond(stream_sid, greeting))


async def feed_pcm_audio(stream_sid: str, pcm_bytes: bytes) -> None:
    sess = speech_sessions.get(stream_sid)
    if not sess or sess.get("closed"):
        return
    try:
        sess["push_stream"].write(pcm_bytes)
    except Exception as e:
        logger.warning("Failed to push audio to Speech STT for %s: %s", stream_sid, e)


async def _on_user_utterance(stream_sid: str, transcript: str) -> None:
    if stream_sid not in speech_sessions:
        return
    sess = speech_sessions.get(stream_sid)
    if not sess or sess.get("closed"):
        return
    if sess.get("bot_speaking") or not sess.get("listening_enabled"):
        _mark_user_activity(stream_sid)
        logger.info(
            "Dropped late STT after listen gate closed: %r stream %s",
            transcript,
            stream_sid,
        )
        return
    ctx = _svc.call_context.get(stream_sid)
    if ctx and ctx.get("status") == "closing":
        return

    sess["last_user_speech_at"] = time.time()
    _cancel_silence_watchdog(stream_sid)
    _disable_listening(stream_sid)

    await _svc.handle_user_service_tag_response(stream_sid, transcript)

    # Reject handlers request their own goodbye turn — do not also reply to "no".
    ctx = _svc.call_context.get(stream_sid)
    if ctx and ctx.get("service_tag_reject_pending_disconnect"):
        return

    prepared = _svc.prepare_user_transcript_for_llm(stream_sid, transcript)
    await inject_and_respond(stream_sid, prepared)


async def inject_and_respond(stream_sid: str, user_text: str) -> None:
    """Append a user (or system-as-user) message, get Nano reply, speak via TTS."""
    sess = speech_sessions.get(stream_sid)
    if not sess or sess.get("closed"):
        return
    ctx = _svc.call_context.get(stream_sid)
    if ctx and ctx.get("status") == "closing":
        return

    async with sess["turn_lock"]:
        sess = speech_sessions.get(stream_sid)
        if not sess or sess.get("closed"):
            return

        if stream_sid in _svc.silence_timer_tasks:
            _cancel_silence_watchdog(stream_sid)

        _disable_listening(stream_sid)

        # Slot step + unmapped STT (e.g. "2122") → re-list slots; never ask yes/no.
        if _svc.should_deterministic_unclear_slot_reask(stream_sid, user_text):
            reply = _svc.build_unclear_slot_reask(stream_sid)
            sess["history"].append(HumanMessage(content=user_text))
            sess["history"].append(AIMessage(content=reply))
            if stream_sid in _svc.call_context:
                _svc.call_context[stream_sid]["last_assistant_message"] = reply
            logger.info(
                "Deterministic unclear-slot reask stream=%s user=%r → %r",
                stream_sid,
                user_text,
                reply,
            )
            speakable = split_speakable_text(reply)
            if speakable:
                await _synthesize_and_queue(stream_sid, speakable)
            else:
                _enable_listening(stream_sid)
                schedule_listening_silence_watchdog(stream_sid)
            return

        sess["history"].append(HumanMessage(content=user_text))
        try:
            ai_msg = await _svc.conversation_llm.ainvoke(sess["history"])
            reply = (ai_msg.content or "").strip()
        except Exception as e:
            logger.error("Nano chat failed for %s: %s", stream_sid, e, exc_info=True)
            sess["history"].pop()
            return

        if not reply:
            return

        reply = _svc.correct_assistant_slot_confirmation(stream_sid, reply)
        reply = _svc.correct_assistant_date_confirmation(stream_sid, reply)
        reply = _svc.align_confirm_reply_to_locked_language(stream_sid, reply)
        reply = _svc.guard_booking_confirmed_reply(stream_sid, reply)

        # Safety net: Nano asked yes/no about slot instead of re-listing
        if _svc.is_in_slot_selection_step(stream_sid) and _svc.is_slot_yes_no_misclarification(
            reply
        ):
            logger.warning(
                "Rewriting slot yes/no misclarification stream=%s was=%r",
                stream_sid,
                reply,
            )
            reply = _svc.build_unclear_slot_reask(stream_sid)

        if _svc.should_force_step2_continuation(stream_sid, reply):
            # Do not speak thanks-only / hold filler; force dates or slots in same turn.
            reply = await _svc.rewrite_incomplete_filler_reply(
                stream_sid, sess["history"], reply
            )
            # rewrite_incomplete_filler_reply already appended messages to history
        else:
            sess["history"].append(AIMessage(content=reply))

        if stream_sid in _svc.call_context:
            _svc.call_context[stream_sid]["last_assistant_message"] = reply
        logger.info("🤖 AI said (Nano): %r for stream %s", reply, stream_sid)

        speakable = split_speakable_text(reply)
        if speakable:
            await _synthesize_and_queue(stream_sid, speakable)
        else:
            _enable_listening(stream_sid)
            schedule_listening_silence_watchdog(stream_sid)

        await _svc.handle_ai_commands(stream_sid, reply)


async def _synthesize_and_queue(
    stream_sid: str,
    text: str,
    *,
    schedule_silence_after: bool = True,
    cancel_silence: bool = True,
) -> None:
    sess = speech_sessions.get(stream_sid)
    if not sess or sess.get("closed"):
        return

    if cancel_silence:
        _cancel_silence_watchdog(stream_sid)
    _disable_listening(stream_sid)

    voice = _pick_tts_voice(stream_sid, text)
    gen = int(sess.get("tts_gen", 0)) + 1
    sess["tts_gen"] = gen
    sess["bot_speaking"] = True
    playback_ev = asyncio.Event()
    sess.setdefault("playback_done_events", {})[gen] = playback_ev
    _svc.response_audio_tracking[stream_sid] = {
        "start_time": time.time(),
        "pcm_bytes": 0,
    }

    def _synth() -> bytes:
        cfg = _speech_config()
        cfg.speech_synthesis_voice_name = voice
        # audio_config=None → audio returned in result
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)
        # Escape XML special chars for SSML text node
        safe = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-IN">'
            f'<voice name="{voice}">{safe}</voice></speak>'
        )
        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            cancel = result.cancellation_details
            raise RuntimeError(
                f"TTS failed: reason={result.reason} details={cancel}"
            )
        return bytes(result.audio_data)

    try:
        pcm = await asyncio.to_thread(_synth)
    except Exception as e:
        logger.error("TTS failed for %s: %s", stream_sid, e, exc_info=True)
        sess["bot_speaking"] = False
        ev = (sess.get("playback_done_events") or {}).get(gen)
        if ev:
            ev.set()
        _svc.response_audio_tracking.pop(stream_sid, None)
        _enable_listening(stream_sid)
        if schedule_silence_after:
            schedule_listening_silence_watchdog(stream_sid)
        return

    sess = speech_sessions.get(stream_sid)
    if not sess or sess.get("closed") or sess.get("tts_gen") != gen:
        logger.info("Discarding stale TTS audio for stream %s", stream_sid)
        return

    if stream_sid in _svc.outbound_audio_buffers:
        _svc.outbound_audio_buffers[stream_sid].extend(pcm)
    tracking = _svc.response_audio_tracking.get(stream_sid)
    audio_duration = len(pcm) / PCM_8K_BYTES_PER_SECOND
    if tracking is not None:
        tracking["pcm_bytes"] = len(pcm)
        tracking["playback_ends_at"] = time.time() + audio_duration

    # Wait for paced send + small grace before opening the listen window.
    generation_time = time.time() - tracking["start_time"] if tracking else 0.0
    remaining_playback = (
        max(0.0, audio_duration - generation_time) + LISTEN_GRACE_AFTER_PLAYBACK_SECONDS
    )

    is_final_closing = bool(
        re.search(r"\bCONFIRMED\b", text, re.I)
        or re.search(r"\b(DECLINE|DECLINED)\b", text, re.I)
        or re.search(r"\bTAG_(SERVICE_TAG_REJECT|RESCHEDULE_DONE)\b", text, re.I)
        or (
            re.search(r"\b(good\s*bye|goodbye|bye)\b|अलविदा|बाय|ವಿದಾಯ|ಬೈ", text, re.I)
            and re.search(
                r"\b(appointment|scheduled|booking|confirming)\b|अपॉइंटमेंट|ಅಪಾಯಿಂಟ್|कन्फर्म|ದೃಢಪಡಿಸ",
                text,
                re.I,
            )
        )
    )
    is_silence_disconnect = bool(
        re.search(
            r"not received a response|कोई जवाब नहीं|ಪ್ರತಿಕ್ರಿಯೆ ಸಿಗಲಿಲ್ಲ",
            text,
            re.I,
        )
    )

    async def _mark_done_and_listen():
        await asyncio.sleep(remaining_playback)
        sess2 = speech_sessions.get(stream_sid)
        if not sess2 or sess2.get("tts_gen") != gen:
            return
        ev = (sess2.get("playback_done_events") or {}).get(gen)
        if ev:
            ev.set()
        ctx = _svc.call_context.get(stream_sid) or {}
        if (
            is_final_closing
            or is_silence_disconnect
            or ctx.get("pending_hangup")
            or ctx.get("status") == "closing"
        ):
            sess2["listening_enabled"] = False
            sess2["bot_speaking"] = False
            logger.info(
                "Playback done — listen stays closed stream=%s gen=%s",
                stream_sid,
                gen,
            )
            return
        _enable_listening(stream_sid)
        if schedule_silence_after:
            schedule_listening_silence_watchdog(stream_sid)
        logger.info(
            "Playback done — listening open stream=%s audio=%.1fs wait=%.1fs silence_after=%s",
            stream_sid,
            audio_duration,
            remaining_playback,
            schedule_silence_after,
        )

    asyncio.create_task(_mark_done_and_listen())


async def stop_speech_session(stream_sid: str) -> None:
    _cancel_silence_watchdog(stream_sid)
    sess = speech_sessions.pop(stream_sid, None)
    if not sess:
        return
    sess["closed"] = True
    sess["tts_gen"] = int(sess.get("tts_gen", 0)) + 1
    try:
        recognizer = sess.get("recognizer")
        if recognizer:
            recognizer.stop_continuous_recognition_async()
    except Exception as e:
        logger.warning("Error stopping Speech recognizer for %s: %s", stream_sid, e)
    try:
        push = sess.get("push_stream")
        if push:
            push.close()
    except Exception as e:
        logger.warning("Error closing Speech push stream for %s: %s", stream_sid, e)
    logger.info("Speech session stopped for stream %s", stream_sid)
