import re
import time
import sys
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketState
import websockets
import os
import httpx
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any, List
import json
import base64
import asyncio
import logging
import audioop
import numpy as np
from scipy import signal
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import calendar

load_dotenv()

import voice_speech_pipeline as speech_pipeline

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# --- Configuration ---
class Config:
    CHUNK_INTERVAL_MS = 100  # 100ms pacing for outbound audio
    CHUNK_BYTES = 1600     # 100ms of 8kHz 16-bit PCM = 1600 bytes


# --- Credentials ---
VOICE = "shimmer"
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_API_TOKEN")
EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN")
EXOTEL_FLOW_APP_ID = os.getenv("EXOTEL_FLOW_APP_ID")
EXOTEL_CALLER_ID = os.getenv("EXOTEL_CALLER_ID")

# Inbound RNNoise: default off; set ENABLE_INBOUND_RNNOISE=1 to enable.
ENABLE_INBOUND_RNNOISE = os.getenv("ENABLE_INBOUND_RNNOISE", "0").lower() in ("1", "true", "yes")
RNNoise = None
if ENABLE_INBOUND_RNNOISE:
    try:
        from pyrnnoise import RNNoise
        logger.info("ENABLE_INBOUND_RNNOISE is enabled — inbound audio uses RNNoise")
    except Exception as e:
        ENABLE_INBOUND_RNNOISE = False
        logger.warning("RNNoise requested but unavailable; continuing without noise suppression: %s", e)


# Azure OpenAI configuration

AZURE_OPENAI_ENDPOINT = "wss://fieldezai.cognitiveservices.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=gpt-realtime"
OPENAI_API_KEY = os.getenv("AZURE_NANO_OPENAI_API_KEY")

# Voice backend: "speech" = Azure Speech STT/TTS + GPT-4.1 Nano; "realtime" = Azure OpenAI Realtime
VOICE_BACKEND = os.getenv("VOICE_BACKEND", "speech").strip().lower()
if VOICE_BACKEND not in ("speech", "realtime"):
    raise RuntimeError('VOICE_BACKEND must be "speech" or "realtime"')

# gpt-4.1-nano
AZURE_NANO_OPENAI_ENDPOINT= os.getenv("AZURE_NANO_OPENAI_ENDPOINT")
AZURE_NANO_OPENAI_API_KEY= os.getenv("AZURE_NANO_OPENAI_API_KEY")
AZURE_NANO_OPENAI_DEPLOYMENT_NAME= os.getenv("AZURE_NANO_OPENAI_DEPLOYMENT_NAME")
AZURE_NANO_OPENAI_API_VERSION= os.getenv("AZURE_NANO_OPENAI_API_VERSION")

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")



# --- Ensure mandatory keys are present ---
_required = [EXOTEL_API_KEY, EXOTEL_API_TOKEN, EXOTEL_SID, EXOTEL_FLOW_APP_ID, EXOTEL_CALLER_ID]
if VOICE_BACKEND == "realtime":
    _required.append(OPENAI_API_KEY)
else:
    _required.extend([
        AZURE_NANO_OPENAI_ENDPOINT,
        AZURE_NANO_OPENAI_API_KEY,
        AZURE_NANO_OPENAI_DEPLOYMENT_NAME,
        AZURE_NANO_OPENAI_API_VERSION,
        AZURE_SPEECH_KEY,
    ])
if not all(_required):
    raise RuntimeError(
        "Please set all required EXOTEL and Azure env vars "
        f"(VOICE_BACKEND={VOICE_BACKEND})."
    )
logger.info("Voice backend configured: %s", VOICE_BACKEND)

class TimeSlot(BaseModel):
    date: str
    slots: List[str] = Field(default_factory=list)
    proximitySlots: List[str] = Field(default_factory=list)
    standardSlots: List[str] = Field(default_factory=list)
    slotOfferMode: str = "STANDARD_ONLY"

class ScheduleCallRequest(BaseModel):
    ticketId: str
    customerPhone: str
    serviceTag: str
    callbackUrl: HttpUrl # Use HttpUrl for validation
    address: str
    pincode: Optional[str] = None
    availableDates: List[TimeSlot]

class PrepareInboundCallRequest(ScheduleCallRequest):
    callSid: str

# Model for the final call report
class CallResult(BaseModel):
    ticketId: str
    callConnected: bool = False
    isLineBusy: bool = False
    slotSelected: bool = False
    selectedDate: Optional[str] = None
    selectedSlot: Optional[str] = None
    comments: Optional[str] = ""
    sentiment: Optional[int] = None
    addressConfirmed: Optional[bool] = None
    serviceTagConfirmed: Optional[bool] = None
    isReschedule: bool = False


# --- In-memory Storage ---
bookings = []
exotel_connections: Dict[str, Dict[str, Any]] = {}
openai_connections: Dict[str, Any] = {}
audio_buffers: Dict[str, bytes] = {}
ai_transcripts: Dict[str, str] = {}
outbound_audio_buffers: Dict[str, bytearray] = {}
sender_tasks: Dict[str, asyncio.Task] = {}
call_context: Dict[str, Dict[str, Any]] = {}
cleanup_locks: Dict[str, asyncio.Lock] = {}
silence_timer_tasks: Dict[str, asyncio.Task] = {}
response_audio_tracking: Dict[str, Dict[str, Any]] = {}
service_tag_reject_fallback_tasks: Dict[str, asyncio.Task] = {}
SILENCE_NUDGE_SECONDS = 5.0
SILENCE_HANGUP_SECONDS = 25.0
SILENCE_TIMEOUT_SECONDS = SILENCE_NUDGE_SECONDS  # legacy alias (realtime path)
LISTEN_GRACE_AFTER_PLAYBACK_SECONDS = 0.4
HANGUP_PLAYBACK_GRACE_SECONDS = 2.5
PCM_8K_BYTES_PER_SECOND = 16000  # 8 kHz * 16-bit mono (Exotel outbound / Speech TTS)


def extract_ticket_id_from_exotel_message(data: dict) -> Optional[str]:
    """Try to read ticketId from Exotel Voicebot `start` (CustomField / custom_parameters vary by version)."""
    st = data.get("start")
    if isinstance(st, dict):
        cp = st.get("custom_parameters") or st.get("customParameters")
        if isinstance(cp, dict):
            for key in ("ticketId", "ticket_id", "TicketId"):
                v = cp.get(key)
                if v is not None and str(v).strip():
                    return str(v).strip()
        for key in ("ticketId", "ticket_id", "TicketId"):
            v = st.get(key)
            if v is not None and str(v).strip():
                return str(v).strip()
        for key in ("custom_field", "CustomField"):
            raw = st.get(key)
            if isinstance(raw, str) and raw.strip():
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        for tk in ("ticketId", "ticket_id"):
                            if obj.get(tk) is not None and str(obj[tk]).strip():
                                return str(obj[tk]).strip()
                except json.JSONDecodeError:
                    pass
    return None


def link_stream_sid_to_call_context(
    stream_sid: str,
    call_sid: str,
    data: dict,
    query_ticket_id: Optional[str] = None,
) -> None:
    """
    Attach initiate-schedule-call payload to this media stream.
    The CallSid on the Voicebot WebSocket often does not match the Sid returned by Calls/connect.json;
    we also register context under ticketId so the start event can still be matched.
    """
    ctx_obj: Optional[Dict[str, Any]] = call_context.pop(call_sid, None)
    tid_ws = extract_ticket_id_from_exotel_message(data)
    if ctx_obj is None and tid_ws:
        ctx_obj = call_context.pop(f"ticket:{tid_ws}", None)
    if ctx_obj is None and query_ticket_id and str(query_ticket_id).strip():
        ctx_obj = call_context.pop(f"ticket:{str(query_ticket_id).strip()}", None)

    if ctx_obj is not None:
        tid = ctx_obj.get("ticketId")
        if tid:
            call_context.pop(f"ticket:{tid}", None)
        for k in list(call_context.keys()):
            if call_context.get(k) is ctx_obj:
                del call_context[k]
        call_context[stream_sid] = ctx_obj
        logger.info(
            "Linked call context stream_sid=%s ticketId=%s exotel_call_sid=%s",
            stream_sid,
            ctx_obj.get("ticketId"),
            call_sid,
        )
        return

    logger.warning(
        "No call context for call_sid=%s ticket_from_ws=%s query_ticket=%s — using UNKNOWN. "
        "start keys: %s",
        call_sid,
        tid_ws,
        query_ticket_id,
        list((data.get("start") or {}).keys()) if isinstance(data.get("start"), dict) else [],
    )
    call_context[stream_sid] = {
        "ticketId": "UNKNOWN",
        "callbackUrl": None,
        "serviceTag": "",
        "serviceTagConfirmed": None,
        "address": "",
        "availableDates": [],
        "addressConfirmed": None,
        "last_assistant_message": "",
        "isReschedule": False,
    }


SERVICE_TAG_REJECT_GOODBYE_USER_PROMPT = (
    "(System notice — the customer has indicated the service tag is NOT correct. "
    "Speak ONE short closing only, in the customer's already-selected language (English, Hindi, or Kannada — "
    "same as the rest of this call). "
    "Include: apologize for the inconvenience; say you cannot continue without verifying the correct service tag; "
    "say our team will get back to them soon (English example: "
    "\"Sorry for the inconvenience. We cannot continue without the correct service tag. "
    "We'll get back to you soon. Goodbye.\" — translate naturally for Hindi or Kannada); end with a brief goodbye. "
    "Do not ask any questions. Do not mention dates or time slots. "
    "On the VERY LAST LINE of your reply only, output exactly TAG_SERVICE_TAG_REJECT (system use only; never speak it aloud)."
)

ULAW_BYTES_PER_SECOND = 8000


def _parse_hhmm_token(part: str) -> Optional[tuple[int, int]]:
    part = part.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", part)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59:
        return None
    return h, mi


def _format_clock_12h(h: int, mi: int) -> str:
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    ap = "AM" if h < 12 else "PM"
    if mi == 0:
        return f"{h12} {ap}"
    return f"{h12}:{mi:02d} {ap}"


def _slot_range_payload_to_spoken_cue(raw: str) -> Optional[str]:
    """
    Map payload like '11:00-14:00' or '09:00–11:00' to '11 AM to 2 PM' (never '11 to 14').
    Returns None if the string is not a simple HH:MM–HH:MM range.
    """
    s = raw.strip()
    for sep in ("–", "—", "−"):
        s = s.replace(sep, "-")
    if "-" not in s:
        return None
    left, _, right = s.partition("-")
    a, b = _parse_hhmm_token(left), _parse_hhmm_token(right)
    if not a or not b:
        return None
    return f"{_format_clock_12h(a[0], a[1])} to {_format_clock_12h(b[0], b[1])}"


def _slot_list_for_prompt(slots: List[str]) -> str:
    """Join slots with explicit 'say as' cues for 24h ranges."""
    parts: List[str] = []
    for raw in slots:
        cue = _slot_range_payload_to_spoken_cue(raw)
        if cue:
            parts.append(f"{raw} (say aloud: {cue})")
        else:
            parts.append(raw)
    return ", ".join(parts) if parts else "(no slots listed)"


_MONTH_HI = {
    1: "जनवरी", 2: "फ़रवरी", 3: "मार्च", 4: "अप्रैल", 5: "मई", 6: "जून",
    7: "जुलाई", 8: "अगस्त", 9: "सितंबर", 10: "अक्टूबर", 11: "नवंबर", 12: "दिसंबर",
}
_MONTH_KN = {
    1: "ಜನವರಿ", 2: "ಫೆಬ್ರವರಿ", 3: "ಮಾರ್ಚ್", 4: "ಏಪ್ರಿಲ್", 5: "ಮೇ", 6: "ಜೂನ್",
    7: "ಜುಲೈ", 8: "ಆಗಸ್ಟ್", 9: "ಸೆಪ್ಟೆಂಬರ್", 10: "ಅಕ್ಟೋಬರ್", 11: "ನವೆಂಬರ್", 12: "ಡಿಸೆಂಬರ್",
}

BOOKING_CONFIRMED_GOODBYE_EN = "Thank you for confirming your appointment. Bye"
BOOKING_CONFIRMED_GOODBYE_HI = "अपॉइंटमेंट कन्फर्म करने के लिए धन्यवाद। बाय।"
BOOKING_CONFIRMED_GOODBYE_KN = "ನಿಮ್ಮ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ದೃಢಪಡಿಸಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ಬೈ."


def detect_locked_language(stream_sid: Optional[str] = None, text: str = "") -> str:
    """Return 'en' | 'hi' | 'kn' from context; text script is fallback only if unlocked."""
    if stream_sid and stream_sid in call_context:
        locked = (call_context[stream_sid].get("lockedLanguage") or "").strip().lower()
        if locked in ("en", "hi", "kn"):
            return locked
    sample = text or ""
    if re.search(r"[\u0C80-\u0CFF]", sample):
        return "kn"
    if re.search(r"[\u0900-\u097F]", sample):
        return "hi"
    return "en"


def locked_language_display_name(lang: str) -> str:
    return {"en": "English", "hi": "Hindi", "kn": "Kannada"}.get(
        (lang or "en").lower(), "English"
    )


def parse_language_choice(transcript: str) -> Optional[str]:
    """
    Detect an explicit language pick / switch request.
    Returns 'en' | 'hi' | 'kn' or None.
    """
    if not transcript:
        return None
    t = transcript.strip().lower()
    # Prefer clear language names (including switch phrasing)
    if re.search(
        r"\b(english|eng)\b|अंग्रेज़[ीि]|ಇಂಗ್ಲಿಷ್|ಇಂಗ್ಲೀಷ್",
        transcript,
        re.I,
    ) or t in ("en", "eng"):
        return "en"
    if re.search(r"\b(hindi|hin)\b|हिंदी|हिन्दी", transcript, re.I) or t in ("hi", "hin"):
        return "hi"
    if re.search(r"\b(kannada|kan|kn)\b|ಕನ್ನಡ", transcript, re.I) or t in ("kn", "kan"):
        return "kn"
    return None


def maybe_lock_language_from_transcript(stream_sid: str, transcript: str) -> None:
    """
    Lock or switch language when the customer clearly names English / Hindi / Kannada.
    Explicit switch mid-call is allowed so lock stays in sync with what the customer wants.
    """
    if not transcript or stream_sid not in call_context:
        return
    choice = parse_language_choice(transcript)
    if not choice:
        return
    ctx = call_context[stream_sid]
    prev = (ctx.get("lockedLanguage") or "").strip().lower()
    if prev == choice:
        return
    ctx["lockedLanguage"] = choice
    if prev in ("en", "hi", "kn"):
        ctx["language_just_switched"] = True
        logger.info(
            "Language switched stream=%s %s → %s (transcript=%r)",
            stream_sid,
            prev,
            choice,
            transcript,
        )
    else:
        logger.info(
            "Language locked stream=%s → %s (transcript=%r)",
            stream_sid,
            choice,
            transcript,
        )


def language_lock_system_hint(stream_sid: str) -> str:
    """Hint appended to user turns so Nano stays in the locked language."""
    lang = detect_locked_language(stream_sid)
    name = locked_language_display_name(lang)
    ctx = call_context.get(stream_sid) or {}
    if ctx.pop("language_just_switched", None):
        return (
            f"[SYSTEM LANGUAGE SWITCH: customer switched to {name}. "
            f"From now on reply ONLY in {name}. Do not mix languages. "
            f"Continue the current step — do not restart greeting or re-ask language.]"
        )
    return (
        f"[SYSTEM LANGUAGE LOCK: customer language is {name}. "
        f"Reply ONLY in {name}. Do not mix languages or switch unless they ask again.]"
    )


def format_date_for_speech(date_str: str, lang: str = "en") -> str:
    """Speak month + day only (no year). Payload remains YYYY-MM-DD internally."""
    if not date_str:
        return ""
    d = str(date_str).replace(":", "-").strip()[:10]
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", d)
    if not m:
        return str(date_str)
    mo, day = int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12):
        return d
    lang = (lang or "en").lower()
    if lang == "hi":
        return f"{day} {_MONTH_HI[mo]}"
    if lang == "kn":
        return f"{_MONTH_KN[mo]} {day}"
    return f"{calendar.month_name[mo]} {day}"


def format_selected_confirm(spoken_value: str, lang: str = "en") -> str:
    """Aligned confirm line across languages: You selected X. Is that correct?"""
    spoken_value = (spoken_value or "").strip()
    lang = (lang or "en").lower()
    if lang == "hi":
        return f"आपने {spoken_value} चुना है। क्या यह सही है?"
    if lang == "kn":
        return f"ನೀವು {spoken_value} ಆಯ್ಕೆ ಮಾಡಿದ್ದೀರಿ. ಇದು ಸರಿಯೇ?"
    return f"You selected {spoken_value}. Is that correct?"


def booking_confirmed_goodbye(lang: str = "en") -> str:
    lang = (lang or "en").lower()
    if lang == "hi":
        return BOOKING_CONFIRMED_GOODBYE_HI
    if lang == "kn":
        return BOOKING_CONFIRMED_GOODBYE_KN
    return BOOKING_CONFIRMED_GOODBYE_EN


def _last_repeatable_prompt(last_assistant_message: str) -> str:
    """Last bot question/line to repeat after a silence nudge."""
    text = re.sub(r"\bTAG_[A-Z0-9_]+\b", "", last_assistant_message or "", flags=re.I)
    text = re.sub(r"\bCONFIRMED\b", "", text, flags=re.I).strip()
    parts = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    if not parts:
        return "Please continue when you are ready."
    for part in reversed(parts):
        if "?" in part:
            return part
    return parts[-1]


def build_still_there_repeat(stream_sid: str) -> str:
    """Deterministic 5s silence nudge: still there + repeat last question."""
    ctx = call_context.get(stream_sid) or {}
    lang = detect_locked_language(stream_sid)
    repeat = _last_repeatable_prompt(ctx.get("last_assistant_message") or "")
    if lang == "hi":
        return f"क्या आप अभी भी वहाँ हैं? {repeat}"
    if lang == "kn":
        return f"ನೀವು ಇನ್ನೂ ಇಲ್ಲಿದ್ದೀರಾ? {repeat}"
    return f"Are you still there? {repeat}"


def build_silence_hangup_message(stream_sid: str) -> str:
    """Spoken line before disconnecting after prolonged silence."""
    lang = detect_locked_language(stream_sid)
    if lang == "hi":
        return "हमें कोई जवाब नहीं मिला। अलविदा।"
    if lang == "kn":
        return "ನಿಮ್ಮಿಂದ ಪ್ರತಿಕ್ರಿಯೆ ಸಿಗಲಿಲ್ಲ. ವಿದಾಯ."
    return "We have not received a response. Goodbye."


# Kannada-script phonetic spellings of English words (common Azure STT output).
_KANNADA_PHONETIC_EN: List[tuple[str, str]] = [
    (r"ಫೋರ್?", "four"),
    (r"ಫೈವ್?", "five"),
    (r"ಸಿಕ್ಸ್?", "six"),
    (r"ಸೆವನ್?", "seven"),
    (r"ಎಯ್ಟ್?", "eight"),
    (r"ನೈನ್?", "nine"),
    (r"ಟೆನ್?", "ten"),
    (r"ಇಲೆವನ್?", "eleven"),
    (r"ಟ್ವೆಲ್ವ್?", "twelve"),
    (r"ಥ್ರೀ|ತ್ರೀ", "three"),
    (r"ಟೂ", "two"),
    (r"ಒನ್?", "one"),
    (r"ಓಕೆ|ಓಕೇ", "ok"),
    (r"ಥ್ಯಾಂಕ್ಸ್?", "thanks"),
    (r"ಯೆಸ್?", "yes"),
    (r"ನೋ", "no"),
    (r"ಇಟ್|ಇಟ್ಸ್?", "it"),
    (r"ಇಸ್", "is"),
    (r"ಕರೆಕ್ಟ್?", "correct"),
    (r"ಇಂಗ್ಲಿಷ್?", "english"),
    (r"ಹಲೋ|ಹೆಲೋ", "hello"),
    (r"ಪಿಎಂ|ಪೀಎಂ", "pm"),
    (r"ಎಎಂ|ಏಎಂ", "am"),
]

_HOUR_WORDS: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def normalize_kannada_phonetic_english(text: str) -> str:
    """Map Kannada-script English phonetics (e.g. ಫೋರ್ ಟು ಸಿಕ್ಸ್) to Latin English."""
    if not text:
        return text
    out = text
    for pattern, repl in _KANNADA_PHONETIC_EN:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    # Standalone ಟು / ಟೂ between tokens → "to" (range connector)
    out = re.sub(r"(?<=\w)\s*ಟು\s*(?=\w)", " to ", out)
    out = re.sub(r"\bಟು\b", "to", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _parse_spoken_hour_token(tok: str) -> Optional[int]:
    tok = tok.strip().lower().replace(".", "")
    if tok in _HOUR_WORDS:
        return _HOUR_WORDS[tok]
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(?:\s*(am|pm))?$", tok)
    if not m:
        return None
    h = int(m.group(1))
    ap = m.group(3)
    if ap == "pm" and h < 12:
        h += 12
    elif ap == "am" and h == 12:
        h = 0
    if 0 <= h <= 23:
        return h
    return None


def _extract_spoken_hour_pair(text: str) -> Optional[tuple[int, int]]:
    """
    Pull (start_hour_24ish, end_hour_24ish) from phrases like 'four to six', '4-6', '4 pm to 6 pm'.
    Hours may still be 1–12 without am/pm; caller resolves against known slots.
    """
    t = normalize_kannada_phonetic_english(text or "").lower()
    t = t.replace("–", "-").replace("—", "-").replace("−", "-")
    t = re.sub(r"\b(o'?clock|hours?)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # four to six / 4 to 6 / 4pm to 6pm / 1:00 pm to 3:00 pm
    m = re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b"
        r"\s*(?:to|-|until|till)\s*"
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
        t,
        re.I,
    )
    if m:
        a = _parse_spoken_hour_token(m.group(1))
        b = _parse_spoken_hour_token(m.group(2))
        if a is not None and b is not None:
            return a, b

    # STT near-miss: "1:23 PM" / "1.23 pm" often means "1 to 3 PM"
    m2 = re.search(r"\b(\d{1,2})[:.](\d{2})\s*(am|pm)?\b", t, re.I)
    if m2:
        a, b = int(m2.group(1)), int(m2.group(2))
        if 1 <= a <= 12 and 1 <= b <= 12:
            ap = (m2.group(3) or "").lower()
            if ap == "pm" and a < 12:
                a = a + 12
            elif ap == "am" and a == 12:
                a = 0
            # keep b in 1–12; matcher expands to PM against afternoon slots
            return a, b

    # Compact digits: "112 to 2" / "123" near "1 to 2" / "1 to 3"
    m3 = re.search(r"\b(\d)\s*(\d)\s*(?:to|-)?\s*(\d{1,2})\s*(am|pm)?\b", t, re.I)
    if m3 and m3.group(1) == m3.group(2):
        # "112 to 2" → ignore duplicate start digit, use 1 and 2
        a, b = int(m3.group(1)), int(m3.group(3))
        if 1 <= a <= 12 and 1 <= b <= 12:
            return a, b
    return None


def _slot_start_end_hours(raw: str) -> Optional[tuple[int, int]]:
    s = raw.strip()
    for sep in ("–", "—", "−"):
        s = s.replace(sep, "-")
    if "-" not in s:
        return None
    left, _, right = s.partition("-")
    a, b = _parse_hhmm_token(left), _parse_hhmm_token(right)
    if not a or not b:
        return None
    return a[0], b[0]


def _hours_match_slot(spoken_a: int, spoken_b: int, slot_start: int, slot_end: int) -> bool:
    """Match 12h spoken hours to 24h slot bounds (prefer PM when ambiguous)."""
    def expand(h: int) -> List[int]:
        h = h % 24
        opts = {h}
        if 1 <= h <= 11:
            opts.add(h + 12)
        if h == 12:
            opts.add(0)
        return list(opts)

    for sa in expand(spoken_a):
        for sb in expand(spoken_b):
            if sa == slot_start and sb == slot_end:
                return True
    return False


def _candidate_hour_pairs_from_text(text: str) -> List[tuple[int, int]]:
    """Collect possible (start, end) hour pairs including STT near-misses."""
    pairs: List[tuple[int, int]] = []
    primary = _extract_spoken_hour_pair(text)
    if primary:
        pairs.append(primary)

    t = normalize_kannada_phonetic_english(text or "").lower()
    t = t.replace("–", "-").replace("—", "-").replace("−", "-")
    t = re.sub(r"\s+", " ", t).strip()

    # STT: "1:23 PM" often means "1 to 3 PM" (colon instead of 'to', digits glued)
    for m2 in re.finditer(r"\b(\d{1,2})[:.](\d{2})\s*(am|pm)?\b", t, re.I):
        a, mm = int(m2.group(1)), int(m2.group(2))
        ap = (m2.group(3) or "").lower()
        starts = [a]
        if ap == "pm" and 1 <= a < 12:
            starts.append(a + 12)
        elif ap == "am" and a == 12:
            starts.append(0)
        ends: List[int] = []
        if 1 <= mm <= 12:
            ends.append(mm)
        else:
            tens, ones = divmod(mm, 10)
            if 1 <= ones <= 12:
                ends.append(ones)
            if 1 <= tens <= 12:
                ends.append(tens)
        for sa in starts:
            for sb in ends:
                pairs.append((sa, sb))
    # de-dupe preserve order
    seen = set()
    out: List[tuple[int, int]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def match_spoken_slot_to_canonical(transcript: str, available_slots: List[str]) -> Optional[str]:
    """
    Map user speech (incl. Kannada-phonetic English) onto a canonical HH:MM-HH:MM slot.
    'four to six' → 16:00-18:00 when that slot exists; never 14:00-16:00.
    Also handles STT misses like '1:23 PM' → 1 PM–3 PM.
    """
    if not available_slots:
        return None
    pairs = _candidate_hour_pairs_from_text(transcript)
    if not pairs:
        return None
    matches: List[str] = []
    for spoken_a, spoken_b in pairs:
        for raw in available_slots:
            bounds = _slot_start_end_hours(str(raw))
            if not bounds:
                continue
            if _hours_match_slot(spoken_a, spoken_b, bounds[0], bounds[1]):
                canon = re.sub(r"[–—‒−]", "-", str(raw).strip())
                if canon not in matches:
                    matches.append(canon)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    afternoon = [m for m in matches if (_slot_start_end_hours(m) or (0, 0))[0] >= 12]
    return afternoon[0] if afternoon else matches[0]


_SLOT_CONFIRM_Q_RE = re.compile(
    r"is that correct|क्या (यह|ये) सही|ಸರಿಯ[ೇೆ]|ಸರಿಯಾಗಿದೆ",
    re.I,
)


def remember_pending_slot_from_text(stream_sid: str, text: str) -> Optional[str]:
    """If text mentions a listed slot (e.g. Nano confirm line), store pendingSelectedSlot."""
    if not text or stream_sid not in call_context:
        return None
    ctx = call_context[stream_sid]
    available = collect_available_slots_for_context(ctx)
    canon = match_spoken_slot_to_canonical(text, available)
    if not canon:
        return None
    prev = ctx.get("pendingSelectedSlot")
    ctx["pendingSelectedSlot"] = canon
    if not ctx.get("pendingSelectedDate"):
        rows = _iter_date_tier_rows(ctx.get("availableDates") or [])
        if len(rows) == 1 and rows[0].get("date"):
            ctx["pendingSelectedDate"] = rows[0]["date"]
        elif ctx.get("confirmedOfferDate"):
            ctx["pendingSelectedDate"] = ctx.get("confirmedOfferDate")
    if prev != canon:
        logger.info(
            "Remembered pending slot stream=%s from text → %s",
            stream_sid,
            canon,
        )
    return canon


def build_slot_choice_prompt(stream_sid: str) -> str:
    """Fixed re-ask when booking goodbye fires without a mapped slot."""
    ctx = call_context.get(stream_sid) or {}
    lang = detect_locked_language(stream_sid)
    available = collect_available_slots_for_context(ctx)
    spoken = []
    for raw in available:
        cue = _slot_range_payload_to_spoken_cue(raw) or raw
        spoken.append(cue)
    slots_line = ", ".join(spoken) if spoken else "the available times"
    if lang == "hi":
        return f"कृपया इनमें से एक स्लॉट चुनें: {slots_line}।"
    if lang == "kn":
        return f"ದಯವಿಟ್ಟು ಈ ಸ್ಲಾಟ್‌ಗಳಿಂದ ಒಂದನ್ನು ಆಯ್ಕೆ ಮಾಡಿ: {slots_line}."
    return f"Please choose one of the available slots: {slots_line}."


def guard_booking_confirmed_reply(stream_sid: str, reply: str) -> str:
    """
    Normalize final goodbye only when date+slot are both pending.
    If Nano tries to CONFIRMED without a mapped slot, re-ask for a slot instead.
    """
    if not reply or stream_sid not in call_context:
        return reply
    if re.search(r"\b(DECLINE|DECLINED|DECLINING)\b", reply, re.I):
        return reply

    # Capture slot from Nano's confirm line before deciding
    if _SLOT_CONFIRM_Q_RE.search(reply) and re.search(
        r"\b(selected|choose|chose|आपने|ಆಯ್ಕೆ)\b", reply, re.I
    ):
        remember_pending_slot_from_text(stream_sid, reply)

    looks_final = bool(
        re.search(r"\bCONFIRMED\b", reply, re.I)
        or _looks_like_final_booking_confirmation(reply)
    )
    if not looks_final:
        return reply

    ctx = call_context[stream_sid]
    date = ctx.get("pendingSelectedDate") or ctx.get("confirmedOfferDate") or ctx.get("selectedDate")
    slot = ctx.get("pendingSelectedSlot") or ctx.get("selectedSlot")
    if date and slot:
        return normalize_confirmed_goodbye(reply, stream_sid)

    if date and not slot:
        ask = build_slot_choice_prompt(stream_sid)
        logger.warning(
            "Blocked premature CONFIRMED goodbye stream=%s date=%s (no pending slot) → re-ask slots",
            stream_sid,
            date,
        )
        return ask

    logger.warning(
        "Blocked premature CONFIRMED goodbye stream=%s (missing date/slot)",
        stream_sid,
    )
    return build_slot_choice_prompt(stream_sid)


def collect_available_slots_for_context(context: Dict[str, Any]) -> List[str]:
    """All slots from payload; prefer slots for confirmed/pending date when known."""
    rows = _iter_date_tier_rows(context.get("availableDates") or [])
    prefer_date = (
        context.get("pendingSelectedDate")
        or context.get("confirmedOfferDate")
        or context.get("selectedDate")
    )
    if prefer_date:
        prefer_date = str(prefer_date).replace(":", "-")[:10]
        for row in rows:
            if row.get("date") == prefer_date:
                return list(row.get("standardSlots") or row.get("slots") or row.get("proximitySlots") or [])
    slots: List[str] = []
    for row in rows:
        for s in (row.get("standardSlots") or row.get("slots") or row.get("proximitySlots") or []):
            if s not in slots:
                slots.append(s)
    return slots


def match_spoken_date_to_canonical(transcript: str, available_dates: List[str]) -> Optional[str]:
    """Map spoken dates like 'July 13th' / 'july thirteenth' onto YYYY-MM-DD from payload."""
    if not transcript or not available_dates:
        return None
    t = normalize_kannada_phonetic_english(transcript).lower()
    t = re.sub(r"\s+", " ", t)
    ordinals = {
        1: r"1st|first", 2: r"2nd|second", 3: r"3rd|third", 4: r"4th|fourth",
        5: r"5th|fifth", 6: r"6th|sixth", 7: r"7th|seventh", 8: r"8th|eighth",
        9: r"9th|ninth", 10: r"10th|tenth", 11: r"11th|eleventh", 12: r"12th|twelfth",
        13: r"13th|thirteenth", 14: r"14th|fourteenth", 15: r"15th|fifteenth",
        16: r"16th|sixteenth", 17: r"17th|seventeenth", 18: r"18th|eighteenth",
        19: r"19th|nineteenth", 20: r"20th|twentieth", 21: r"21st|twenty[-\s]?first",
        22: r"22nd|twenty[-\s]?second", 23: r"23rd|twenty[-\s]?third",
        24: r"24th|twenty[-\s]?fourth", 25: r"25th|twenty[-\s]?fifth",
        26: r"26th|twenty[-\s]?sixth", 27: r"27th|twenty[-\s]?seventh",
        28: r"28th|twenty[-\s]?eighth", 29: r"29th|twenty[-\s]?ninth",
        30: r"30th|thirtieth", 31: r"31st|thirty[-\s]?first",
    }
    months = {
        1: r"jan(?:uary)?", 2: r"feb(?:ruary)?", 3: r"mar(?:ch)?", 4: r"apr(?:il)?",
        5: r"may", 6: r"jun(?:e)?", 7: r"jul(?:y)?", 8: r"aug(?:ust)?",
        9: r"sep(?:t(?:ember)?)?", 10: r"oct(?:ober)?", 11: r"nov(?:ember)?", 12: r"dec(?:ember)?",
    }
    for raw in available_dates:
        d = str(raw).replace(":", "-")[:10]
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", d)
        if not m:
            continue
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if d in t.replace("/", "-"):
            return d
        mon = months.get(month)
        day_pat = ordinals.get(day) or str(day)
        if mon and re.search(rf"\b{mon}\b", t) and re.search(rf"\b(?:{day_pat}|{day})\b", t):
            return d
    return None


def prepare_user_transcript_for_llm(stream_sid: str, transcript: str) -> str:
    """
    Post-process STT: Kannada-phonetic → English, map spoken ranges to canonical slots,
    and inject an unambiguous hint for Nano.
    """
    if not transcript or stream_sid not in call_context:
        return transcript
    maybe_lock_language_from_transcript(stream_sid, transcript)
    ctx = call_context[stream_sid]
    lang_hint = language_lock_system_hint(stream_sid)
    normalized = normalize_kannada_phonetic_english(transcript)
    rows = _iter_date_tier_rows(ctx.get("availableDates") or [])
    date_list = [r["date"] for r in rows if r.get("date")]

    matched_date = match_spoken_date_to_canonical(normalized, date_list)
    if matched_date:
        ctx["pendingSelectedDate"] = matched_date
        ctx["confirmedOfferDate"] = matched_date
    elif _classify_yes_no_confirmation(normalized) == "positive":
        last = ctx.get("last_assistant_message") or ""
        if re.search(
            r"\b(date|selected|july|january|february|march|april|may|june|august|september|october|november|december|would like)\b|"
            r"आपने|चुना|तारीख|जुलाई|ನೀವು|ಆಯ್ಕೆ|ದಿನಾಂಕ|ಜುಲೈ",
            last,
            re.I,
        ):
            from_last = match_spoken_date_to_canonical(last, date_list)
            if from_last:
                ctx["pendingSelectedDate"] = from_last
                ctx["confirmedOfferDate"] = from_last
        # Yes after "You selected 1 PM to 3 PM. Is that correct?" → lock that slot
        if _SLOT_CONFIRM_Q_RE.search(last):
            remember_pending_slot_from_text(stream_sid, last)

    available = collect_available_slots_for_context(ctx)
    canonical = match_spoken_slot_to_canonical(normalized, available)
    if not canonical:
        base = normalized if normalized != transcript else transcript
        return f"{base} {lang_hint}"

    spoken = _slot_range_payload_to_spoken_cue(canonical) or canonical
    ctx["pendingSelectedSlot"] = canonical
    if not ctx.get("pendingSelectedDate"):
        if len(rows) == 1 and rows[0].get("date"):
            ctx["pendingSelectedDate"] = rows[0]["date"]
        elif ctx.get("confirmedOfferDate"):
            ctx["pendingSelectedDate"] = ctx.get("confirmedOfferDate")

    enriched = (
        f'{normalized}. '
        f'[SYSTEM SLOT MAP: user chose canonical slot {canonical} '
        f'(spoken "{spoken}"). '
        f'This means ONLY that window. '
        f'"four to six" / "4 to 6" is 4 PM to 6 PM (16:00-18:00), NEVER 2 PM to 4 PM (14:00-16:00). '
        f'Confirm this exact slot with the customer.] '
        f'{lang_hint}'
    )
    logger.info(
        "Slot map stream=%s transcript=%r normalized=%r → %s",
        stream_sid,
        transcript,
        normalized,
        canonical,
    )
    return enriched


def _reply_script_lang(reply: str) -> Optional[str]:
    if not reply:
        return None
    if re.search(r"[\u0C80-\u0CFF]", reply):
        return "kn"
    if re.search(r"[\u0900-\u097F]", reply):
        return "hi"
    if re.search(r"[A-Za-z]", reply):
        return "en"
    return None


def align_confirm_reply_to_locked_language(stream_sid: str, reply: str) -> str:
    """
    If Nano asked a date/slot confirm in the wrong language vs lockedLanguage,
    rewrite to the locked-language confirm line (stops EN↔KN flip loops).
    """
    if not reply or stream_sid not in call_context:
        return reply
    if not _SLOT_CONFIRM_Q_RE.search(reply) and not re.search(
        r"is that correct|क्या (यह|ये) सही|ಸರಿಯ[ೇೆ]",
        reply,
        re.I,
    ):
        return reply
    if re.search(r"\b(CONFIRMED|DECLINE|TAG_)\b", reply, re.I):
        return reply
    locked = detect_locked_language(stream_sid)
    script = _reply_script_lang(reply)
    if script == locked:
        return reply

    ctx = call_context[stream_sid]
    pending_slot = ctx.get("pendingSelectedSlot")
    pending_date = ctx.get("pendingSelectedDate") or ctx.get("confirmedOfferDate")

    # Prefer slot confirm when a slot was mentioned / pending
    available = collect_available_slots_for_context(ctx)
    mentioned_slot = match_spoken_slot_to_canonical(reply, available) or pending_slot
    if mentioned_slot and re.search(
        r"\b(?:AM|PM|a\.m\.|p\.m\.|\d{1,2}:\d{2}|to)\b|सुबह|दोपहर|ಬೆಳಿಗ್ಗೆ|ಮಧ್ಯಾಹ್ನ",
        reply,
        re.I,
    ):
        spoken = _slot_range_payload_to_spoken_cue(mentioned_slot) or mentioned_slot
        fixed = format_selected_confirm(spoken, locked)
        logger.info(
            "Aligned slot confirm language stream=%s %s→%s %r",
            stream_sid,
            script,
            locked,
            fixed,
        )
        return fixed

    rows = _iter_date_tier_rows(ctx.get("availableDates") or [])
    date_list = [r["date"] for r in rows if r.get("date")]
    mentioned_date = match_spoken_date_to_canonical(reply, date_list) or pending_date
    if mentioned_date:
        spoken = format_date_for_speech(mentioned_date, locked)
        fixed = format_selected_confirm(spoken, locked)
        logger.info(
            "Aligned date confirm language stream=%s %s→%s %r",
            stream_sid,
            script,
            locked,
            fixed,
        )
        return fixed
    return reply


def correct_assistant_slot_confirmation(stream_sid: str, reply: str) -> str:
    """If Nano confirms a neighboring slot but we already mapped the user pick, rewrite."""
    if not reply or stream_sid not in call_context:
        return reply
    ctx = call_context[stream_sid]
    # Always capture Nano's spoken confirm slot into pending when we can map it
    if _SLOT_CONFIRM_Q_RE.search(reply):
        remember_pending_slot_from_text(stream_sid, reply)
    pending = ctx.get("pendingSelectedSlot")
    if not pending:
        return reply
    available = collect_available_slots_for_context(ctx)
    mentioned = match_spoken_slot_to_canonical(reply, available)
    if not mentioned or mentioned == pending:
        return reply
    if not re.search(
        r"\b(selected|choose|chose|is that correct|confirm)\b|क्या (यह|ये) सही|ಸರಿಯ[ೇೆ]|ಸರಿಯಾಗಿದೆ",
        reply,
        re.I,
    ):
        return reply
    lang = detect_locked_language(stream_sid)
    spoken = _slot_range_payload_to_spoken_cue(pending) or pending
    fixed = format_selected_confirm(spoken, lang)
    logger.info(
        "Corrected wrong slot confirm stream=%s nano=%s pending=%s → %r",
        stream_sid,
        mentioned,
        pending,
        fixed,
    )
    return fixed


def correct_assistant_date_confirmation(stream_sid: str, reply: str) -> str:
    """Force simple date confirm aligned across EN/HI/KN."""
    if not reply or stream_sid not in call_context:
        return reply
    if re.search(r"\breschedule\b|पुनः\s*शेड्यूल|ಮರುನಿಗದಿ", reply, re.I):
        return reply
    if not re.search(
        r"is that correct|क्या (यह|ये) सही|ಸರಿಯ[ೇೆ]|ಸರಿಯಾಗಿದೆ",
        reply,
        re.I,
    ):
        return reply
    # Slot / booking turns — leave alone
    if re.search(r"\b(CONFIRMED|DECLINE)\b", reply, re.I):
        return reply
    if re.search(
        r"\b(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*"
        r"(?:AM|PM|a\.m\.|p\.m\.)\b|"
        r"\b(?:सुबह|दोपहर|शाम|रात|ಬೆಳಿಗ್ಗೆ|ಮಧ್ಯಾಹ್ನ|ಸಂಜೆ)\b",
        reply,
        re.I,
    ):
        return reply
    ctx = call_context[stream_sid]
    pending = ctx.get("pendingSelectedDate") or ctx.get("confirmedOfferDate")
    rows = _iter_date_tier_rows(ctx.get("availableDates") or [])
    date_list = [r["date"] for r in rows if r.get("date")]
    if not pending:
        pending = match_spoken_date_to_canonical(reply, date_list)
    if not pending:
        return reply
    # Only short date-confirm questions
    if len(reply) > 180:
        return reply
    if not re.search(
        r"\b(would like|you selected|you chose|selected)\b|आपने|चुना|ಆಯ್ಕೆ|ನೀವು",
        reply,
        re.I,
    ) and not match_spoken_date_to_canonical(reply, date_list):
        return reply
    lang = detect_locked_language(stream_sid)
    spoken = format_date_for_speech(pending, lang)
    if not spoken:
        return reply
    fixed = format_selected_confirm(spoken, lang)
    if re.sub(r"[\s—–-]+", " ", reply).strip().lower() == re.sub(r"[\s—–-]+", " ", fixed).lower():
        return reply
    logger.info(
        "Normalized date confirm stream=%s nano=%r → %r",
        stream_sid,
        reply,
        fixed,
    )
    return fixed


_HOLD_FILLER_RE = re.compile(
    r"\b("
    r"please\s+hold|please\s+wait|hold\s+on|hold\s+the\s+line|"
    r"one\s+moment|just\s+a\s+(?:moment|second|sec)|"
    r"let\s+me\s+check|while\s+i\s+check|checking\s+(?:the\s+)?available|"
    r"wait\s+a\s+moment|i(?:'| a)?m\s+checking|give\s+me\s+a\s+(?:moment|second)"
    r")\b",
    re.I,
)

_SCHEDULING_CONTENT_RE = re.compile(
    r"\b("
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"जानवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर|"
    r"ಜನವರಿ|ಫೆಬ್ರವರಿ|ಮಾರ್ಚ್|ಏಪ್ರಿಲ್|ಮೇ|ಜೂನ್|ಜುಲೈ|ಆಗಸ್ಟ್|ಸೆಪ್ಟೆಂಬರ್|ಅಕ್ಟೋಬರ್|ನವೆಂಬರ್|ಡಿಸೆಂಬರ್|"
    r"reschedule|रीशेड्यूल|ಮರುನಿಗದಿ|"
    r"time\s+slot|available\s+(?:time\s+)?slots?|"
    r"\d{1,2}\s*(?:AM|PM|a\.m\.|p\.m\.)"
    r")\b",
    re.I,
)

_SERVICE_TAG_THANKS_ONLY_RE = re.compile(
    r"("
    r"thanks?\s+for\s+confirming\s+(?:the\s+)?service\s*tag|"
    r"thank\s+you\s+for\s+confirming\s+(?:the\s+)?service\s*tag|"
    r"सर्विस\s*टैग\s*(?:कन्फर्म|पुष्टि)|"
    r"ಸರ್ವೀಸ್\s*ಟ್ಯಾಗ್\s*ದೃಢಪಡಿಸ"
    r")",
    re.I,
)

_FORCE_LIST_DATES_PROMPT = (
    "(System correction — your previous reply thanked for the service tag or stalled, "
    "but did NOT list appointment dates/slots. "
    "Reply again NOW in the customer's locked language. In ONE turn: briefly thank them for the service tag if needed, "
    "then immediately continue Step 2 — list EVERY date (or slots in SINGLE_DATE_MODE) from the canonical schedule "
    "as month and day only (no year), then offer reschedule when in MULTIPLE_DATE_MODE. "
    "FORBIDDEN: ending after thanks only; please hold; please wait; hold on; let me check. "
    "Do NOT speak the year. Do NOT re-ask the service tag. Do NOT ask about address.)"
)


def reply_has_scheduling_content(reply: str) -> bool:
    """True if the reply already lists dates and/or time slots."""
    if not reply:
        return False
    if _SCHEDULING_CONTENT_RE.search(reply):
        return True
    # Spoken day like "July 15" / "15 जुलाई" often enough with month already covered;
    # also catch HH:MM-HH:MM style
    if re.search(r"\b\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\b", reply):
        return True
    return False


def is_incomplete_scheduling_filler(reply: str) -> bool:
    """True when Nano stalls ('please hold…') instead of listing dates/slots."""
    if not reply or not _HOLD_FILLER_RE.search(reply):
        return False
    if reply_has_scheduling_content(reply):
        return False
    return True


def is_incomplete_service_tag_ack(stream_sid: str, reply: str) -> bool:
    """
    True when Nano only thanks for service-tag confirm and stops —
    must continue with dates/slots in the same turn.
    """
    if not reply or stream_sid not in call_context:
        return False
    ctx = call_context[stream_sid]
    if ctx.get("serviceTagConfirmed") is not True:
        return False
    # Already past tag → date/slot booking path
    if ctx.get("slotSelected") or ctx.get("pendingSelectedSlot"):
        return False
    if re.search(r"\b(CONFIRMED|DECLINE|TAG_)\b", reply, re.I):
        return False
    if reply_has_scheduling_content(reply):
        return False
    # Explicit thanks-for-tag without calendar content
    if _SERVICE_TAG_THANKS_ONLY_RE.search(reply):
        return True
    # Short post-confirm ack with no schedule content (Nano stopped early)
    stripped = re.sub(r"\s+", " ", reply).strip()
    if len(stripped) <= 140 and re.search(
        r"\b(thank|thanks|धन्यवाद|ಧನ್ಯವಾದ)\b",
        stripped,
        re.I,
    ):
        return True
    return False


def should_force_step2_continuation(stream_sid: str, reply: str) -> bool:
    """Hold-filler OR thanks-only after service tag — force Step 2 list."""
    return is_incomplete_scheduling_filler(reply) or is_incomplete_service_tag_ack(
        stream_sid, reply
    )


def build_deterministic_step2_after_tag(stream_sid: str) -> str:
    """Fixed thanks + dates/slots from payload (no Nano)."""
    ctx = call_context.get(stream_sid) or {}
    rows = _iter_date_tier_rows(ctx.get("availableDates") or [])
    lang = detect_locked_language(stream_sid)

    if not rows:
        if lang == "hi":
            return (
                "सर्विस टैग कन्फर्म करने के लिए धन्यवाद। अभी सिस्टम में कोई स्लॉट उपलब्ध नहीं है। "
                "हमारी टीम जल्द संपर्क करेगी।"
            )
        if lang == "kn":
            return (
                "ಸರ್ವೀಸ್ ಟ್ಯಾಗ್ ದೃಢಪಡಿಸಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ಈಗ ಸಿಸ್ಟಂನಲ್ಲಿ ಸ್ಲಾಟ್ ಲಭ್ಯವಿಲ್ಲ. "
                "ನಮ್ಮ ತಂಡ ಶೀಘ್ರದಲ್ಲೇ ಸಂಪರ್ಕಿಸುತ್ತದೆ."
            )
        return (
            "Thanks for confirming the service tag. No appointment slots are available in the system right now. "
            "Our team will reach out soon."
        )

    # Single date → go straight to slots
    if len(rows) == 1:
        row = rows[0]
        d = row.get("date") or ""
        spoken_date = format_date_for_speech(d, lang)
        slots = row.get("proximitySlots") or row.get("standardSlots") or row.get("slots") or []
        spoken_slots = []
        for raw in slots:
            cue = _slot_range_payload_to_spoken_cue(str(raw)) or str(raw)
            spoken_slots.append(cue)
        slots_line = ", ".join(spoken_slots) if spoken_slots else "the available times"
        if lang == "hi":
            return (
                f"सर्विस टैग कन्फर्म करने के लिए धन्यवाद। {spoken_date} के लिए उपलब्ध समय स्लॉट हैं: "
                f"{slots_line}। कृपया एक चुनें।"
            )
        if lang == "kn":
            return (
                f"ಸರ್ವೀಸ್ ಟ್ಯಾಗ್ ದೃಢಪಡಿಸಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. {spoken_date} ಗೆ ಲಭ್ಯವಿರುವ ಸಮಯ ಸ್ಲಾಟ್‌ಗಳು: "
                f"{slots_line}. ದಯವಿಟ್ಟು ಒಂದನ್ನು ಆಯ್ಕೆ ಮಾಡಿ."
            )
        return (
            f"Thanks for confirming the service tag. For {spoken_date}, the available time slots are: "
            f"{slots_line}. Please choose one."
        )

    spoken_dates = []
    for row in rows:
        d = row.get("date") or ""
        if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            spoken_dates.append(format_date_for_speech(d, lang))
    dates_line = ", ".join(spoken_dates) if spoken_dates else "the available dates"
    if lang == "hi":
        return (
            f"सर्विस टैग कन्फर्म करने के लिए धन्यवाद। कृपया इनमें से एक तारीख चुनें: "
            f"{dates_line}। या रीशेड्यूल।"
        )
    if lang == "kn":
        return (
            f"ಸರ್ವೀಸ್ ಟ್ಯಾಗ್ ದೃಢಪಡಿಸಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ದಯವಿಟ್ಟು ಈ ದಿನಾಂಕಗಳಿಂದ ಒಂದನ್ನು ಆಯ್ಕೆ ಮಾಡಿ: "
            f"{dates_line}. ಅಥವಾ ಮರುನಿಗದಿ."
        )
    return (
        f"Thanks for confirming the service tag. Please choose an appointment date from the following options: "
        f"{dates_line}. Or reschedule."
    )


async def rewrite_incomplete_filler_reply(stream_sid: str, history: list, filler_reply: str) -> str:
    """Force Step 2 (dates/slots) when Nano stalled or stopped after service-tag thanks."""
    logger.warning(
        "Incomplete Step-2 reply for stream %s — forcing date/slot list. Was: %r",
        stream_sid,
        filler_reply,
    )
    # Prefer deterministic script immediately for thanks-only (most reliable)
    if is_incomplete_service_tag_ack(stream_sid, filler_reply):
        reply = build_deterministic_step2_after_tag(stream_sid)
        history.append(AIMessage(content=filler_reply))
        history.append(AIMessage(content=reply))
        logger.info("Used deterministic Step-2 after service-tag thanks stream=%s", stream_sid)
        return reply

    history.append(AIMessage(content=filler_reply))
    history.append(HumanMessage(content=_FORCE_LIST_DATES_PROMPT))
    try:
        ai_msg = await conversation_llm.ainvoke(history)
        reply = (ai_msg.content or "").strip()
    except Exception as e:
        logger.error("Force date-list rewrite failed for %s: %s", stream_sid, e, exc_info=True)
        reply = ""
    if (
        not reply
        or is_incomplete_scheduling_filler(reply)
        or is_incomplete_service_tag_ack(stream_sid, reply)
        or not reply_has_scheduling_content(reply)
    ):
        reply = build_deterministic_step2_after_tag(stream_sid)
        logger.info("Used deterministic Step-2 fallback for stream %s", stream_sid)
    history.append(AIMessage(content=reply))
    return reply


def _looks_like_final_booking_confirmation(message: str) -> bool:
    if not message:
        return False
    low = message.lower()
    has_bye = bool(
        re.search(r"\b(good\s*bye|goodbye|bye)\b|ಅಲವಿದ|अलविदा|ವಿದಾಯ|बाय|ಬೈ", low, re.I)
    )
    has_appt = bool(
        re.search(
            r"\b(appointment|scheduled|booking|confirmed|confirming your appointment)\b|"
            r"अपॉइंटमेंट|ಅಪಾಯಿಂಟ್ಮೆಂಟ್|कन्फर्म|ದೃಢಪಡಿಸ",
            low,
            re.I,
        )
    )
    return has_bye and has_appt


def normalize_confirmed_goodbye(reply: str, stream_sid: Optional[str] = None) -> str:
    """Force the short booking goodbye; keep CONFIRMED on its own last line for the system."""
    if not reply:
        return reply
    if re.search(r"\b(DECLINE|DECLINED|DECLINING)\b", reply, re.I):
        return reply
    if not (
        re.search(r"\bCONFIRMED\b", reply, re.I)
        or _looks_like_final_booking_confirmation(reply)
    ):
        return reply
    # Avoid rewriting mid-flow "thanks for confirming the service tag"
    if re.search(r"service\s*tag|सर्विस\s*टैग|ಸರ್ವೀಸ್\s*ಟ್ಯಾಗ್", reply, re.I) and not re.search(
        r"\b(CONFIRMED|goodbye|good\s*bye|bye)\b|अलविदा|बाय|ವಿದಾಯ|ಬೈ",
        reply,
        re.I,
    ):
        return reply
    lang = detect_locked_language(stream_sid)
    return f"{booking_confirmed_goodbye(lang)}\nCONFIRMED"


def _force_booking_fields_from_context(stream_sid: str, message: str) -> None:
    """When Nano says goodbye booking but omits CONFIRMED / NLU fails, fill report from pending slot."""
    if stream_sid not in call_context:
        return
    ctx = call_context[stream_sid]
    slot = ctx.get("pendingSelectedSlot") or ctx.get("selectedSlot")
    date = ctx.get("pendingSelectedDate") or ctx.get("confirmedOfferDate") or ctx.get("selectedDate")
    # Try pull 12h range from assistant text and map
    if not slot:
        available = collect_available_slots_for_context(ctx)
        slot = match_spoken_slot_to_canonical(message, available)
    if date:
        date = str(date).replace(":", "-")
        if re.match(r"^\d{4}-\d{2}-\d{2}", date):
            date = date[:10]
        else:
            # leave as-is; NLU may have used YYYY:MM:DD
            date = str(date).replace(":", "-")[:10] if re.search(r"\d{4}", str(date)) else date
    if slot and date:
        ctx["slotSelected"] = True
        ctx["selectedDate"] = date
        ctx["selectedSlot"] = re.sub(r"[–—‒−]", "-", str(slot))
        ctx["comments"] = ""
        logger.info(
            "Forced booking fields stream=%s date=%s slot=%s",
            stream_sid,
            ctx["selectedDate"],
            ctx["selectedSlot"],
        )


def _normalize_day_entry(item: Any) -> Dict[str, Any]:
    """Normalize one availableDates row to tier fields."""
    if isinstance(item, TimeSlot):
        proximity = [str(s) for s in (item.proximitySlots or [])]
        standard = [str(s) for s in (item.standardSlots or [])]
        flat = [str(s) for s in (item.slots or [])]
        mode = (item.slotOfferMode or "").strip() or (
            "PROXIMITY_FIRST" if proximity else "STANDARD_ONLY"
        )
        if not standard and not proximity and flat:
            standard = flat
            mode = "STANDARD_ONLY"
        return {
            "date": str(item.date).strip(),
            "proximitySlots": proximity,
            "standardSlots": standard,
            "slotOfferMode": mode,
            "slots": flat or (proximity + standard),
        }
    if isinstance(item, dict):
        proximity = [str(s) for s in (item.get("proximitySlots") or [])]
        standard = [str(s) for s in (item.get("standardSlots") or [])]
        flat = [str(s) for s in (item.get("slots") or [])]
        mode = (item.get("slotOfferMode") or "").strip() or (
            "PROXIMITY_FIRST" if proximity else "STANDARD_ONLY"
        )
        if not standard and not proximity and flat:
            standard = flat
            mode = "STANDARD_ONLY"
        return {
            "date": str(item.get("date", "")).strip(),
            "proximitySlots": proximity,
            "standardSlots": standard,
            "slotOfferMode": mode,
            "slots": flat or (proximity + standard),
        }
    if hasattr(item, "date"):
        return _normalize_day_entry(
            TimeSlot(
                date=str(getattr(item, "date")),
                slots=list(getattr(item, "slots", None) or []),
                proximitySlots=list(getattr(item, "proximitySlots", None) or []),
                standardSlots=list(getattr(item, "standardSlots", None) or []),
                slotOfferMode=str(getattr(item, "slotOfferMode", None) or "STANDARD_ONLY"),
            )
        )
    return {"date": "", "proximitySlots": [], "standardSlots": [], "slotOfferMode": "STANDARD_ONLY", "slots": []}


def _iter_available_date_rows(available_dates_obj: Any) -> List[tuple[str, List[str]]]:
    """Normalize payload `availableDates` to (YYYY-MM-DD, [slot strings]) — flat union."""
    rows: List[tuple[str, List[str]]] = []
    for item in available_dates_obj or []:
        norm = _normalize_day_entry(item)
        if norm["date"]:
            rows.append((norm["date"], norm["slots"]))
    return rows


def _iter_date_tier_rows(available_dates_obj: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in available_dates_obj or []:
        norm = _normalize_day_entry(item)
        if norm["date"]:
            rows.append(norm)
    return rows


def build_scheduling_calendar_prompt_parts(available_dates_obj: Any) -> tuple[str, str, str, str]:
    """
    Build prompt injections for multi-date scheduling with optional proximity-first tiers.
    Returns (available_dates_summary, scheduling_mode_instructions, scheduled_date_fallback, available_slots_fallback).
    """
    tier_rows = _iter_date_tier_rows(available_dates_obj)
    if not tier_rows:
        summary = "  (No appointment dates were provided in the system payload.)"
        mode = (
            "NO_DATES_MODE: No valid dates were provided. After service tag confirmation, apologize briefly that no slots "
            "are available in the system and say our team will reach out; do not invent dates or times."
        )
        return summary, mode, "No date", ""

    lines: List[str] = []
    any_proximity = False
    for row in tier_rows:
        d = row["date"]
        mode = row.get("slotOfferMode") or "STANDARD_ONLY"
        prox = row.get("proximitySlots") or []
        std = row.get("standardSlots") or []
        if prox:
            any_proximity = True
            prox_part = _slot_list_for_prompt(prox)
            std_part = _slot_list_for_prompt(std) if std else "(none)"
            lines.append(
                f"  - {d} [PROXIMITY_FIRST] (canonical YYYY-MM-DD):\n"
                f"      proximity_slots (offer FIRST for this date only): {prox_part}\n"
                f"      standard_fallback_slots (offer ONLY after customer declines ALL proximity_slots for this same date): {std_part}"
            )
        else:
            slot_part = _slot_list_for_prompt(std or row.get("slots") or [])
            lines.append(
                f"  - {d} [STANDARD_ONLY] (canonical YYYY-MM-DD): slots {slot_part}"
            )
    summary = "\n".join(lines)

    if len(tier_rows) == 1:
        row0 = tier_rows[0]
        d0 = row0["date"]
        if row0.get("proximitySlots"):
            prox_list = _slot_list_for_prompt(row0["proximitySlots"])
            std_list = _slot_list_for_prompt(row0.get("standardSlots") or [])
            mode = (
                "SINGLE_DATE_PROXIMITY_FIRST_MODE: There is exactly ONE appointment date. After thanks for confirming "
                f"the service tag, offer ONLY proximity_slots for {d0}: {prox_list}. "
                "Say they are recommended for the customer's location. Ask if either works. "
                "If the customer declines ALL proximity slots, immediately offer standard_fallback_slots for the SAME date "
                f"({d0}) only: {std_list}. Do NOT jump to another date. "
                "Only valid bookings are slots from the active tier the customer accepted. "
                "Speak slots using '(say aloud: …)' twelve-hour wording. "
                "Do NOT ask the customer to confirm the service address."
            )
            return summary, mode, d0, prox_list

        slots0 = row0.get("standardSlots") or row0.get("slots") or []
        slot_list = _slot_list_for_prompt(slots0) if slots0 else ""
        mode = (
            "SINGLE_DATE_MODE: There is exactly ONE appointment date in the system list. After you thank the customer for "
            "confirming the service tag, do NOT ask them to choose among several dates. Go directly to time-slot selection "
            f"for date {d0} only. The only valid slots for that date are: {slot_list}. "
            "Speak each slot using the '(say aloud: …)' twelve-hour wording; never read raw hour numbers like '11 to 14'. "
            "Do NOT ask the customer to confirm the service address."
        )
        return summary, mode, d0, slot_list

    proximity_block = ""
    if any_proximity:
        proximity_block = (
            "\nPROXIMITY-FIRST RULES (apply per confirmed date):\n"
            "1. After the customer confirms a date (PATH A), if that date row is marked [PROXIMITY_FIRST], offer ONLY "
            "proximity_slots first with wording like: 'We have the following appointment slots recommended for your "
            "location on [spoken date]:' then list proximity_slots.\n"
            "2. If the customer declines ALL proximity_slots for that date, immediately offer standard_fallback_slots "
            "for the SAME date — do NOT offer another date yet.\n"
            "3. Only after the customer declines BOTH proximity and standard_fallback for that date, ask if they want "
            "to try another listed date.\n"
            "4. For dates marked [STANDARD_ONLY], offer standard slots directly after date confirmation.\n"
            "5. Never mix proximity_slots from one date with slots from another date in the same turn.\n"
        )

    mode = (
        "MULTIPLE_DATE_MODE: After you thank the customer for confirming the service tag, in the SAME turn ask them to choose an appointment DATE. "
        "List **every** date from the canonical list above in spoken form (month and day only, no year, in their locked language). "
        "Immediately after listing those dates, also offer a **reschedule option** with the short line only "
        "(English: \"Or reschedule.\" / Hindi: \"या रीशेड्यूल।\" / Kannada: \"ಅಥವಾ ಮರುನಿಗದಿ.\"). "
        "Do NOT use the longer 'if none of these dates work' wording. "
        "Do NOT say please hold, please wait, let me check, or any stalling phrase — the dates are already in the system list. "
        "Do NOT read time slots until either (A) a listed date is chosen and confirmed, or (B) the customer chooses reschedule. "
        "Do NOT ask the customer to confirm the service address.\n"
        "PATH A — Pick a listed date: When they indicate one of the system dates, map to exactly one YYYY-MM-DD row. "
        "Confirm with the SAME simple line in the locked language — English: \"You selected [spoken date — month and day only, no year]. Is that correct?\" "
        "Hindi: \"आपने [spoken date] चुना है। क्या यह सही है?\" "
        "Kannada: \"ನೀವು [spoken date] ಆಯ್ಕೆ ಮಾಡಿದ್ದೀರಿ. ಇದು ಸರಿಯೇ?\" "
        "Only after clear YES, follow proximity-first "
        "rules below for that date's tier, then list ALL slots for that date in the same turn (never say 'please wait'), "
        "then slot confirmation and booking flow as usual.\n"
        "PATH B — Reschedule: If they say they need to reschedule / none of these work / similar, first confirm intent "
        "(English: \"You would like us to reschedule — is that correct?\" / "
        "Hindi: \"आप रीशेड्यूल चुनना चाहते हैं — क्या यह सही है?\" / "
        "Kannada: \"ನೀವು ಮರುನಿಗದಿ ಆಯ್ಕೆ ಮಾಡುತ್ತಿದ್ದೀರಿ — ಇದು ಸರಿಯೇ?\"). "
        "Only after they clearly say YES: thank them for confirming, say you will note the reschedule request, "
        "that the team will get back to them soon, and goodbye. Do NOT book a slot. "
        "On the VERY LAST LINE only, output exactly TAG_RESCHEDULE_DONE (system use; never speak it aloud)."
        + proximity_block
    )
    first_d = tier_rows[0]["date"]
    first_slots = tier_rows[0].get("proximitySlots") or tier_rows[0].get("standardSlots") or tier_rows[0].get("slots") or []
    return summary, mode, first_d, ", ".join(first_slots)


_EMPTY_ADDRESS_LABEL_RE = re.compile(
    r"^(city|state|pin\s*code|pincode|zip|country|district)\s*:\s*$",
    re.IGNORECASE,
)
_TRAILING_EMPTY_ADDRESS_LABEL_RE = re.compile(
    r"\s+(City|State|Pin\s*Code|Pincode|Zip|Country|District)\s*:\s*$",
    re.IGNORECASE,
)
_PINCODE_RE = re.compile(r"^\d{6}$")
_PINCODE_LABEL_RE = re.compile(r"^(?:pin\s*code|pincode|zip)\s*:\s*(\d{6})\s*$", re.IGNORECASE)
_CITY_LABEL_RE = re.compile(r"^city\s*:\s*(.+)$", re.IGNORECASE)
_ORG_RE = re.compile(
    r"\b(private limited|pvt\.?\s*ltd|limited|ltd\.?|llp|inc|corp|corporation|services)\b",
    re.IGNORECASE,
)
_BUILDING_RE = re.compile(
    r"\b(tower|building|block|wing|solarium|campus|complex|plaza|centre|center|tech\s*park|it\s*park|bagmane|argon)\b",
    re.IGNORECASE,
)
_LOCALITY_RE = re.compile(
    r"\b(road|street|lane|village|hobli|nagar|layout|cross|sector|phase|plot|area|main)\b",
    re.IGNORECASE,
)
_FLOOR_ONLY_RE = re.compile(r"^floors?\s*#?\d", re.IGNORECASE)


def _address_part_key(part: str) -> str:
    return re.sub(r"\s+", " ", part.strip().lower())


def _parse_address_parts(address: str) -> List[str]:
    parts = re.split(r"[,;]", str(address).strip())
    cleaned: List[str] = []
    seen: set[str] = set()
    for part in parts:
        segment = re.sub(r"\s+", " ", part.strip())
        if not segment or _EMPTY_ADDRESS_LABEL_RE.match(segment):
            continue
        segment = _TRAILING_EMPTY_ADDRESS_LABEL_RE.sub("", segment).strip()
        if not segment or _EMPTY_ADDRESS_LABEL_RE.match(segment):
            continue
        key = _address_part_key(segment)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(segment)
    return cleaned


def _classify_address_part(part: str, *, is_first: bool) -> str:
    pin_match = _PINCODE_LABEL_RE.match(part)
    if pin_match:
        return "pincode"
    if _PINCODE_RE.match(part):
        return "pincode"
    city_match = _CITY_LABEL_RE.match(part)
    if city_match and city_match.group(1).strip():
        return "city"
    if _FLOOR_ONLY_RE.match(part):
        return "skip"
    if is_first and _ORG_RE.search(part):
        return "company"
    if _BUILDING_RE.search(part) and not _LOCALITY_RE.search(part):
        return "building"
    if _LOCALITY_RE.search(part):
        return "locality"
    if is_first:
        return "company"
    if _BUILDING_RE.search(part):
        return "building"
    return "other"


def summarize_address_for_speech(address: str) -> str:
    """Short phone readback: company, locality, building, city, pincode (max 5 parts)."""
    if not address or not str(address).strip():
        return "Not provided."
    parts = _parse_address_parts(address)
    if not parts:
        return str(address).strip()

    company: List[str] = []
    locality: List[str] = []
    building: List[str] = []
    city: List[str] = []
    pincode: List[str] = []
    other: List[str] = []

    for i, part in enumerate(parts):
        kind = _classify_address_part(part, is_first=(i == 0))
        if kind == "skip":
            continue
        if kind == "company":
            company.append(part)
        elif kind == "locality":
            locality.append(part)
        elif kind == "building":
            building.append(part)
        elif kind == "city":
            city_match = _CITY_LABEL_RE.match(part)
            city.append(city_match.group(1).strip() if city_match else part)
        elif kind == "pincode":
            pin_match = _PINCODE_LABEL_RE.match(part)
            pincode.append(pin_match.group(1) if pin_match else part)
        else:
            other.append(part)

    if not company and other:
        company.append(other.pop(0))

    segments: List[str] = []
    if company:
        segments.append(company[0])
    if locality:
        segments.append(", ".join(locality))
    if building:
        segments.append(", ".join(building))
    if city:
        segments.append(city[0])
    if pincode:
        segments.append(pincode[0])

    if not segments:
        return ", ".join(parts[:5])
    return ", ".join(segments[:5])


###################################### 2025 -09 -10 evening prompt ###############################################
SYSTEM_PROMPT_TEMPLATE1 = """
CORE DIRECTIVES
OPENING AND LANGUAGE SELECTION:
At the start of the conversation, speak Step 0a and Step 0b in English first, in order, before asking for language.
Then ask the customer to select their preferred language from: English, Hindi, or Kannada.
If the input is unclear, background noise, or not one of the supported languages, DO NOT auto-select.
Politely ask the customer to repeat:
👉 “I’m sorry, I didn’t catch that. Could you please say English, Hindi, or Kannada?”
Once a supported language is clearly detected (e.g. the customer says “English”, “Hindi”, or “Kannada”), lock in that language immediately and proceed to Step 1 — do NOT ask “You selected [Language]. Is that correct?” or any other confirmation.
Do not proceed to the service tag step until a supported language is clearly detected.

STRICTLY FORBIDDEN:
NO MIXED LANGUAGES: Under NO circumstances are you to use any other language after the customer's language preference is locked in. This rule applies to all prompts and all dynamic data. For example, if Hindi is selected, dates and times MUST be spoken only in Hindi, not a mix of Hindi and English.
NO INTERNAL DATA: Never speak or reference internal system commands, JSON data, sentiment scores, booking commands, numbers used for internal purposes, hangup commands, or tokens such as TAG_SERVICE_TAG_REJECT or TAG_RESCHEDULE_DONE.
DO NOT READ BRACKETS: Never speak or read aloud any text inside curly braces {} or square brackets []. These are internal system placeholders or instructions, not part of the script to be spoken to the customer.

PERSONA:
You are a friendly and efficient AI scheduling assistant for FieldEZ. Your tone should be clear, friendly, and natural, not robotic.

STRICT SELECTION RULE:
Every valid appointment date and time range comes **only** from the canonical system list below. You must not invent, change, or merge dates or slots. Do not choose a date or time slot on behalf of the customer. If you are unsure what they chose, ask again. Do not assume.

TIME SLOT SPEECH (mandatory):
Payloads may show windows like `11:00-14:00` (24-hour). You MUST speak them in **twelve-hour AM/PM** using the `(say aloud: …)` cue next to each slot in the system list (e.g. **11 AM to 2 PM**, not "11 to 14"). **Never** read only the hour digits as two numbers (e.g. never "nine to eleven" for `09:00-11:00` in the wrong style, and never "eleven to fourteen"). In Hindi/Kannada, express the same twelve-hour meaning naturally.

SLOT DISAMBIGUATION (critical):
If the customer says **four to six** / **4 to 6** / **4 PM to 6 PM** (or Kannada-phonetic STT like ಫೋರ್ ಟು ಸಿಕ್ಸ್), that means **ONLY 4 PM to 6 PM** (canonical `16:00-18:00` when listed). It is **NEVER** 2 PM to 4 PM (`14:00-16:00`). Do not confuse the end of one slot with the start of the next. When a [SYSTEM SLOT MAP] hint appears in the user message, trust that canonical slot exactly.
If STT is unclear, ask them to repeat the slot using AM/PM (e.g. “Did you mean 4 PM to 6 PM?”). Do not guess a neighboring window.

BACKGROUND NOISE HANDLING:
If the response is unclear, garbled, or nonsensical, DO NOT guess. Politely ask them to repeat in their selected language.

YOUR TASK
Ticket ID for this call: {{ticket_id}}
Service tag for this call (read exactly as given — speak slowly and clearly): {{service_tag}}
Do NOT ask the customer to confirm or verify the service address. Address verification is not part of this call.

{{scheduling_mode_instructions}}

Canonical schedule from the system (never offer dates or slots that are not listed here):
{{available_dates_summary}}

MANDATORY CONVERSATION FLOW
Step 0 (OPENING — speak in this exact order before anything else)

Step 0a (INTRO — always speak in English):
👉 “Hello! I am calling from Dell scheduling regarding your service appointment.”

Step 0b (RECORDING NOTICE — always speak in English immediately after Step 0a):
👉 “This call will be recorded for training and quality purposes.”

Step 0c (LANGUAGE SELECTION — after Step 0a and 0b):
👉 “To better assist you, please select your preferred language: English, Hindi, or Kannada.”
If unclear/noise/invalid → repeat the language request only (do not repeat 0a/0b unless the call restarted).
Once a supported language is clearly detected → lock it in immediately and proceed to Step 1. Do NOT confirm with “You selected [Language]. Is that correct?”

Step 1 (SERVICE TAG CONFIRMATION)
Immediately after language is locked in, speak **in the customer’s selected language** using this structure (for English, follow it closely; for other languages, translate the same meaning naturally):
👉 “Let’s confirm the service tag for your service.
Your service tag is: {{service_tag}}.
Please confirm — is this correct?”
Read the service tag **slowly and clearly** — pause briefly between characters, digits, and symbols. Do not rush. Read {{service_tag}} exactly as provided; do not translate or change it regardless of the customer’s selected language.
If the customer asks to repeat, say “pardon”, “again”, “didn’t hear”, or similar → repeat the **same** service tag slowly and ask again: “Please confirm — is this correct?” Do **not** proceed to date/slot steps yet.
If the response is unclear, garbled, partial, or ambiguous → do **not** guess. Politely ask them to repeat. Only proceed when you are **100%** sure they said yes or no.
Rules:
If the customer says NO, wrong, incorrect, not correct, or clearly rejects the service tag → apologize briefly, say you cannot continue without the correct service tag, say our team will get back to them soon, say goodbye, and **end the conversation**. Do NOT ask for dates or times.
If you must disconnect for a wrong service tag, put the exact token TAG_SERVICE_TAG_REJECT alone on the very last line of your response (system use only; do not speak this token aloud).
If you receive a system notice that the service tag was rejected, follow it exactly: speak that closing apology and callback promise in the customer's language, then TAG_SERVICE_TAG_REJECT on the last line only.
If the customer says YES, correct, right, or clearly confirms the service tag → acknowledge with a short thanks for confirming the service tag **in their selected language** (English example: “Thanks for confirming the service tag.”), **then** proceed to Step 2 (date and time) per **{{scheduling_mode_instructions}}**. Do **not** ask about the service address.
If unclear or noise → ask them to repeat. Do not assume. Do not proceed until confirmation is clear.

Step 2 (DATE AND TIME — obey {{scheduling_mode_instructions}})

PROXIMITY_FIRST (per date — when the canonical list marks a date [PROXIMITY_FIRST]):
2p-a — After date is confirmed (or immediately in SINGLE_DATE_PROXIMITY_FIRST_MODE), offer ONLY proximity_slots for that date. Say they are recommended for the customer's location (translate naturally). Example (English):
👉 "We have the following appointment slots available on [spoken date] that are recommended for your location: [list proximity_slots in 12-hour AM/PM]. If either of these slots is convenient, we can schedule your appointment accordingly. Otherwise, please let me know, and I'll be happy to suggest additional available slots."
2p-b — If the customer declines ALL proximity_slots (not convenient / no / other slots / equivalent), immediately offer standard_fallback_slots for the **same** date only. Example (English):
👉 "No problem. For [spoken date], these additional slots are available: [list standard_fallback_slots in 12-hour AM/PM]. Please choose one."
2p-c — Do NOT move to another date until both proximity and standard_fallback tiers are declined for the current date (MULTIPLE_DATE_MODE only).
2p-d — Valid booking slots are only from the tier the customer ultimately accepts (proximity or standard).

SINGLE_DATE_PROXIMITY_FIRST_MODE:
Follow PROXIMITY_FIRST rules 2p-a and 2p-b for the only date — no date-selection step.

SINGLE_DATE_MODE:
After thanks for confirming the service tag, go straight to time slots for the **only** date in the canonical list. In the customer’s language (English example; translate for Hindi or Kannada):
👉 “We can schedule your appointment. For [spoken form of that date], the available time slots are: [list **only** valid slots for that date in 12-hour AM/PM]. Please choose one.”
If unclear → ask them to repeat. Do not auto-select a slot. After they choose, confirm: “You selected [Time Slot]. Is that correct?” Only after YES → continue toward Step 4.

MULTIPLE_DATE_MODE:
2a — After thanks for confirming the service tag, ask them to **select an appointment date**, list **only** the dates from the canonical list (spoken in full form), **then** add one more option: if they need **reschedule** instead because none of those dates work (see {{scheduling_mode_instructions}}). Do not read time slots yet.
2b — **Pick a listed date:** match their choice to one YYYY-MM-DD row. Confirm: “You selected [spoken date] — is that correct?” Only after clear YES continue.
2c — In the **same turn** after date YES, offer **all** slots for that confirmed date (never say “please wait a moment” or pause without listing them). Confirm slot with YES before proceeding.
2d — **Reschedule path:** If they want reschedule, confirm (English): “You would like us to reschedule — is that correct?” After clear YES: thank them, say you will reschedule and the team will get back to them soon, goodbye. Last line only: TAG_RESCHEDULE_DONE (never spoken). End the call; do not offer slots.

NO_DATES_MODE:
After service tag confirmation, explain politely that no schedule was loaded and our team will follow up; do not invent slots.

Shared: Never offer slots for a date the customer has not confirmed in MULTIPLE_DATE_MODE.

Step 3 (BOOKING CONFIRMATION - INTERNAL CHECK)
(Internal Check - Do Not Speak)
You MUST NOT proceed to Step 4 until:
1. Confirmed service tag from Step 1.
2. MULTIPLE_DATE_MODE: customer has **confirmed with YES** which appointment date (one YYYY-MM-DD from the list) before slots were offered for that date.
3. SINGLE_DATE_MODE: the single system date is the appointment date for slot selection.
4. A clear **YES** for the chosen **time slot** after you asked “Is that correct?” for the slot.

Formatting Rule for Step 4:
[Spoken Date] = the appointment date being booked (the only date in SINGLE_DATE_MODE, or the date the customer confirmed in MULTIPLE_DATE_MODE). Say it as **Month Day only** (no year) in their language.
[Spoken Time] = chosen slot in 12-hour AM/PM (e.g., “9 AM to 11 AM”) in their language.
Raw codes and internal JSON are for system use only and must not be read aloud.

Action: When all conditions above are met, proceed immediately to Step 4.

Step 4 (FINAL TURN)
✅ If the customer CONFIRMS the booking, speak ONLY this short goodbye (translate naturally for Hindi/Kannada). Do NOT repeat date/time again. On the VERY LAST LINE only, output exactly CONFIRMED (system use; never speak it aloud).
English (speak this exactly):
"Thank you for confirming your appointment. Bye"
Then last line only: CONFIRMED
Hindi (speak): "अपॉइंटमेंट कन्फर्म करने के लिए धन्यवाद। बाय।" then last line: CONFIRMED
Kannada (speak): "ನಿಮ್ಮ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ದೃಢಪಡಿಸಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ಬೈ." then last line: CONFIRMED

When confirming the **slot** choice (before final booking), repeat the slot in 12-hour AM/PM exactly as listed (e.g. “You selected 4 PM to 6 PM. Is that correct?”). Never swap to a different window.

❌ If the customer DECLINES the booking, strictly respond (in their selected language):
English:
"Thank you for your time. I understand that you would like to DECLINE the booking for now, and thank you for sharing your { "comments": "<comments>" }. If you change your mind, please don't hesitate to call us for rescheduling. Wishing you a great day! Good Bye!..."
Hindi:
"आपके समय के लिए धन्यवाद। मैं समझता हूँ कि आप अभी के लिए बुकिंग DECLINE करना चाहते हैं, और आपके { "comments": "<comments>" } साझा करने के लिए धन्यवाद। अगर आप अपना मन बदलते हैं, तो कृपया हमें कॉल करके पुनः शेड्यूल करें। आपको शुभ दिन की शुभकामनाएँ! अलविदा..."
Kannada:
"ನಿಮ್ಮ ಸಮಯಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ನೀವು ಈಗಾಗಲೇ ಬುಕ್ಕಿಂಗ್ DECLINE ಮಾಡಲು ಬಯಸುತ್ತೀರಿ ಎಂದು ನಾನು ಅರ್ಥಮಾಡಿಕೊಂಡಿದ್ದೇನೆ, ಮತ್ತು ನಿಮ್ಮ { "comments": "<comments>" } ಹಂಚಿಕೊಂಡಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ನಿಮ್ಮ ಅಭಿಪ್ರಾಯ ಬದಲಾದರೆ, ದಯವಿಟ್ಟು ನಮಗೆ ಕರೆ ಮಾಡಿ ಮರುನಿಗದಿಪಡಿಸಿಕೊಳ್ಳಿ. ನಿಮಗೆ ಶುಭ ದಿನವಾಗಲಿ! ವಿದಾಯ..."

ABSOLUTE RULES
Do NOT accept or process any input while you are speaking. Always finish your full prompt first.
Ignore background noise while speaking.
Always confirm service tag, date, and time before finalizing. Do NOT ask for address confirmation.
Never cut your prompt short. Always complete full sentences.
Add a natural half-second pause at the end of each prompt before listening.
"""

###################################### 2025 -07 -13 evening prompt ###############################################
SYSTEM_PROMPT_TEMPLATE = """
CORE DIRECTIVES
OPENING AND LANGUAGE SELECTION:
At the start of the conversation, speak Step 0a and Step 0b in English first, in order, before asking for language.
Then ask the customer to select their preferred language from: English, Hindi, or Kannada.
If the input is unclear, background noise, or not one of the supported languages, DO NOT auto-select.
Politely ask the customer to repeat:
👉 “I’m sorry, I didn’t catch that. Could you please say English, Hindi, or Kannada?”
Once a supported language is clearly detected (e.g. the customer says “English”, “Hindi”, or “Kannada”), lock in that language immediately and proceed to Step 1 — do NOT ask “You selected [Language]. Is that correct?” or any other confirmation.
Do not proceed to Step 1 until a supported language is clearly detected.

STRICTLY FORBIDDEN:
NO MIXED LANGUAGES: After the customer's language is locked, reply ONLY in that language. If they clearly ask to switch (e.g. “English”, “switch to English”, “Hindi”, “Kannada”), switch immediately and continue the current step in the new language — do not restart the greeting. Never mix languages in one turn. For example, if English is locked, dates and confirms MUST be English only.
NO INTERNAL DATA: Never speak or reference internal system commands, JSON data, sentiment scores, booking commands, numbers used for internal purposes, hangup commands, or tokens such as TAG_SERVICE_TAG_REJECT, TAG_ADDRESS_REJECT or TAG_RESCHEDULE_DONE.
DO NOT READ BRACKETS: Never speak or read aloud any text inside curly braces {} or square brackets []. These are internal system placeholders or instructions, not part of the script to be spoken to the customer.
NO FILLER / HOLD TURNS: Never say “please hold”, “please wait”, “hold on”, “one moment”, “let me check”, “checking the available dates/slots”, or any similar stalling line. You already have the canonical schedule below — speak it immediately in the same turn. Never end a turn without completing the required list (dates or slots).

PERSONA:
You are a friendly and efficient AI scheduling assistant for FieldEZ. Your tone should be clear, friendly, and natural, not robotic.

STRICT SELECTION RULE:
Every valid appointment date and time range comes **only** from the canonical system list below. You must not invent, change, or merge dates or slots. Do not choose a date or time slot on behalf of the customer. If you are unsure what they chose, ask again. Do not assume.

TIME SLOT SPEECH (mandatory):
Payloads may show windows like `11:00-14:00` (24-hour). You MUST speak them in **twelve-hour AM/PM** using the `(say aloud: …)` cue next to each slot in the system list (e.g. **11 AM to 2 PM**, not "11 to 14"). **Never** read only the hour digits as two numbers (e.g. never "nine to eleven" for `09:00-11:00` in the wrong style, and never "eleven to fourteen"). In Hindi/Kannada, express the same twelve-hour meaning naturally.

SPOKEN DATE RULE (mandatory):
Canonical dates are stored as YYYY-MM-DD in the system list. When **speaking** or **confirming** a date to the customer, say **only month and day** (no year) in the locked language:
English: “July 13” / “July thirteenth”
Hindi: “13 जुलाई”
Kannada: “ಜುಲೈ 13”
Do **NOT** speak the year (never “two thousand twenty-six”, never “2026”). Do **NOT** ask the customer to confirm the year. Internally always map their choice back to the matching YYYY-MM-DD row (year is already known from the payload).

LANGUAGE PHRASE ALIGNMENT (use these exact patterns after language lock):
Date confirm — English: “You selected [date]. Is that correct?” | Hindi: “आपने [date] चुना है। क्या यह सही है?” | Kannada: “ನೀವು [date] ಆಯ್ಕೆ ಮಾಡಿದ್ದೀರಿ. ಇದು ಸರಿಯೇ?”
Slot confirm — English: “You selected [time slot]. Is that correct?” | Hindi: “आपने [time slot] चुना है। क्या यह सही है?” | Kannada: “ನೀವು [time slot] ಆಯ್ಕೆ ಮಾಡಿದ್ದೀರಿ. ಇದು ಸರಿಯೇ?”
Booking goodbye — English: “Thank you for confirming your appointment. Bye” | Hindi: “अपॉइंटमेंट कन्फर्म करने के लिए धन्यवाद। बाय।” | Kannada: “ನಿಮ್ಮ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ದೃಢಪಡಿಸಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ಬೈ.” Then last line only: CONFIRMED (never speak CONFIRMED).
Do not invent longer goodbye lines. Do not repeat date/time again on the final goodbye.

BACKGROUND NOISE HANDLING:
If the response is unclear, garbled, or nonsensical, DO NOT guess. Politely ask them to repeat in their selected language.

YOUR TASK
Ticket ID for this call: {{ticket_id}}
Service tag for this call (read exactly as given — speak slowly and clearly): {{service_tag}}
Do NOT ask the customer to confirm or verify the service address. Address verification is not part of this call.

{{scheduling_mode_instructions}}

Canonical schedule from the system (never offer dates or slots that are not listed here):
{{available_dates_summary}}

MANDATORY CONVERSATION FLOW
Step 0 (OPENING — speak in this exact order before anything else)

Step 0a (INTRO — always speak in English):
👉 “Hello, I am calling from Dell Scheduling.
This call is recorded for quality purposes”

Step 0b (LANGUAGE SELECTION — after Step 0a):
👉 “Please choose English, Hindi or Kannada.”
If unclear/noise/invalid → repeat the language request only (do not repeat 0a unless the call restarted).
Once a supported language is clearly detected → lock it in immediately and proceed to Step 1. Do NOT confirm with “You selected [Language]. Is that correct?”

Step 1 (SERVICE TAG CONFIRMATION)
Immediately after language is locked in, speak **in the customer’s selected language** using this structure:
English: “Your service tag is: {{service_tag}}. Please confirm — is this correct?”
Hindi: “आपका सर्विस टैग है: {{service_tag}}. कृपया पुष्टि करें — क्या यह सही है?”
Kannada: “ನಿಮ್ಮ ಸರ್ವೀಸ್ ಟ್ಯಾಗ್: {{service_tag}}. ದಯವಿಟ್ಟು ದೃಢಪಡಿಸಿ — ಇದು ಸರಿಯೇ?”
Read the service tag **slowly and clearly** — pause briefly between characters, digits, and symbols (e.g. letters, numbers, dashes). Do not rush. Read {{service_tag}} exactly as provided; do not translate or change it regardless of the customer’s selected language.
If the customer asks to repeat, say “pardon”, “again”, “didn’t hear”, or similar → repeat the **same** service tag slowly and ask again with the same confirm question in the locked language. Do **not** proceed to date/slot steps yet.
If the response is unclear, garbled, partial, or ambiguous (not a clear yes or no) → do **not** guess. Politely ask them to repeat:
English: “I didn’t catch that clearly. Please say yes if the service tag is correct, or no if it is wrong.”
Hindi: “मुझे साफ़ सुनाई नहीं दिया। सर्विस टैग सही हो तो हाँ कहें, गलत हो तो ना कहें।”
Kannada: “ಸ್ಪಷ್ಟವಾಗಿ ಕೇಳಿಸಲಿಲ್ಲ. ಸರ್ವೀಸ್ ಟ್ಯಾಗ್ ಸರಿಯಾಗಿದ್ದರೆ ಹೌದು, ತಪ್ಪಾಗಿದ್ದರೆ ಇಲ್ಲ ಎಂದು ಹೇಳಿ.”
Only proceed when you are **100%** sure they said yes or no.
Rules:
If the customer says NO, wrong, incorrect, not correct, or clearly rejects the service tag → apologize briefly, say you cannot continue without the correct service tag, say our team will get back to them soon, say goodbye, and **end the conversation**. Do NOT ask for dates or times.
If you must disconnect for a wrong service tag, put the exact token TAG_SERVICE_TAG_REJECT alone on the very last line of your response (system use only; do not speak this token aloud).
If you receive a system notice that the service tag was rejected, follow it exactly: speak that closing apology and callback promise in the customer's language, then TAG_SERVICE_TAG_REJECT on the last line only.
If the customer says YES, correct, right, or clearly confirms the service tag → in the **SAME turn**: (1) short thanks for confirming the service tag in the locked language
(English: “Thanks for confirming the service tag.” / Hindi: “सर्विस टैग कन्फर्म करने के लिए धन्यवाद।” / Kannada: “ಸರ್ವೀಸ್ ಟ್ಯಾಗ್ ದೃಢಪಡಿಸಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು.”),
then (2) immediately continue with Step 2 (list dates or slots per mode). Do **not** stop after thanks. Do **not** say please hold / please wait / let me check. Do **not** ask about the service address.
If unclear or noise → ask them to repeat. Do not assume. Do not proceed until confirmation is clear.

Step 2 (DATE AND TIME — obey {{scheduling_mode_instructions}})

SINGLE_DATE_MODE:
After confirming the Service Tag, in the **same turn**, go straight to time slots for the **only** date in the canonical list:
English: “The available time slots are: [list **only** valid slots for that date in 12-hour AM/PM]. Please choose one.”
Hindi: “उपलब्ध समय स्लॉट हैं: [list]. कृपया एक चुनें।”
Kannada: “ಲಭ್ಯವಿರುವ ಸಮಯ ಸ್ಲಾಟ್‌ಗಳು: [list]. ದಯವಿಟ್ಟು ಒಂದನ್ನು ಆಯ್ಕೆ ಮಾಡಿ.”
If unclear → ask them to repeat. Do not auto-select a slot. After they choose, confirm with LANGUAGE PHRASE ALIGNMENT slot confirm. Only after YES → continue toward Step 4.

MULTIPLE_DATE_MODE:
2a — After confirming the Service Tag, in the **same turn**, ask them to choose a date, **list every date** from the canonical list in spoken form in their locked language (no year), then offer reschedule with the short line only:
English: “Or reschedule.”
Hindi: “या रीशेड्यूल।”
Kannada: “ಅಥವಾ ಮರುನಿಗದಿ.”
Do NOT use longer wording like “if none of these dates work”. Do not read time slots yet. Never say please hold / please wait.
2b — **Pick a listed date:** match their choice to one YYYY-MM-DD row. Confirm with LANGUAGE PHRASE ALIGNMENT date confirm. Only after clear YES continue.
   If they only say yes/correct/ok **without naming a date** right after you listed dates, ask which date they want — do **not** treat that as reschedule and do **not** re-ask the service tag.
2c — In the **same turn** after date YES, offer **all** slots for that confirmed date (never say “please wait”). Confirm slot with LANGUAGE PHRASE ALIGNMENT slot confirm before proceeding.
2d — **Reschedule path:** Only if they clearly ask to reschedule / none of the dates work. Confirm:
English: “You choose to reschedule — is that correct?”
Hindi: “आप रीशेड्यूल चुनना चाहते हैं — क्या यह सही है?”
Kannada: “ನೀವು ಮರುನಿಗದಿ ಆಯ್ಕೆ ಮಾಡುತ್ತಿದ್ದೀರಿ — ಇದು ಸರಿಯೇ?”
If unclear or ambiguous → do **not** guess. Ask them to repeat yes or no until you are **100%** sure. If they say **NO** → go back to Step 2a and offer the listed dates again; do not end the call and do not DECLINE.
Only after clear **YES** to reschedule intent, ask:
English: “Which date would you like to reschedule?”
Hindi: “आप किस तारीख के लिए रीशेड्यूल चाहेंगे?”
Kannada: “ಯಾವ ದಿನಾಂಕಕ್ಕೆ ಮರುನಿಗದಿ ಬೇಕು?”
If the date answer is unclear → ask them to repeat. If they cannot specify a date, note that politely and continue to closing.
After you have their preferred date (or they cannot specify one): thank them, repeat the preferred date in spoken form if given, say the team will get back to them soon, goodbye. Last line only: TAG_RESCHEDULE_DONE (never spoken). End the call; do not offer slots.

NO_DATES_MODE:
After service tag confirmation, explain politely that no schedule was loaded and our team will follow up; do not invent slots.

Shared: Never offer slots for a date the customer has not confirmed in MULTIPLE_DATE_MODE.
If the customer says “hello” / “are you there” after you already confirmed the service tag, do **not** restart Step 1 — continue from the current Step 2 question (repeat dates or slots as needed).

Step 3 (BOOKING CONFIRMATION - INTERNAL CHECK)
(Internal Check - Do Not Speak)
You MUST NOT proceed to Step 4 until:
1. Confirmed service tag from Step 1.
2. MULTIPLE_DATE_MODE: customer has **confirmed with YES** which appointment date (one YYYY-MM-DD from the list) before slots were offered for that date.
3. SINGLE_DATE_MODE: the single system date is the appointment date for slot selection.
4. A clear **YES** for the chosen **time slot** after you asked “Is that correct?” for the slot.

Formatting Rule for Step 3:
[Spoken Date] = the appointment date being booked (the only date in SINGLE_DATE_MODE, or the date the customer confirmed in MULTIPLE_DATE_MODE). Say it as **Month Day only** (no year) in their language.
[Spoken Time] = chosen slot in 12-hour AM/PM (e.g., “9 AM to 11 AM”) in their language.
Raw codes and internal JSON are for system use only and must not be read aloud.

Action: When all conditions above are met, proceed immediately to Step 4.

Step 4 (FINAL TURN)
✅ If the customer CONFIRMS the booking, speak ONLY the short goodbye from LANGUAGE PHRASE ALIGNMENT. Do NOT repeat date/time again. On the VERY LAST LINE only, output exactly CONFIRMED (system use; never speak it aloud).
English (speak this exactly):
"Thank you for confirming your appointment. Bye"
Then last line only: CONFIRMED
Hindi (speak exactly):
"अपॉइंटमेंट कन्फर्म करने के लिए धन्यवाद। बाय।"
Then last line only: CONFIRMED
Kannada (speak exactly):
"ನಿಮ್ಮ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ದೃಢಪಡಿಸಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ಬೈ."
Then last line only: CONFIRMED

❌ If the customer DECLINES the booking, strictly respond (in their selected language):
English:
"Thank you for your time. I understand that you would like to DECLINE the booking for now, and thank you for sharing your { "comments": "<comments>" }. If you change your mind, please don't hesitate to call us for rescheduling. Wishing you a great day! Good Bye!..."
Hindi:
"आपके समय के लिए धन्यवाद। मैं समझता हूँ कि आप अभी के लिए बुकिंग DECLINE करना चाहते हैं, और आपके { "comments": "<comments>" } साझा करने के लिए धन्यवाद। अगर आप अपना मन बदलते हैं, तो कृपया हमें कॉल करके पुनः शेड्यूल करें। आपको शुभ दिन की शुभकामनाएँ! अलविदा..."
Kannada:
"ನಿಮ್ಮ ಸಮಯಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ನೀವು ಈಗಾಗಲೇ ಬುಕ್ಕಿಂಗ್ DECLINE ಮಾಡಲು ಬಯಸುತ್ತೀರಿ ಎಂದು ನಾನು ಅರ್ಥಮಾಡಿಕೊಂಡಿದ್ದೇನೆ, ಮತ್ತು ನಿಮ್ಮ { "comments": "<comments>" } ಹಂಚಿಕೊಂಡಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ನಿಮ್ಮ ಅಭಿಪ್ರಾಯ ಬದಲಾದರೆ, ದಯವಿಟ್ಟು ನಮಗೆ ಕರೆ ಮಾಡಿ ಮರುನಿಗದಿಪಡಿಸಿಕೊಳ್ಳಿ. ನಿಮಗೆ ಶುಭ ದಿನವಾಗಲಿ! ವಿದಾಯ..."

ABSOLUTE RULES
Do NOT accept or process any input while you are speaking. Always finish your full prompt first.
Ignore background noise while speaking.
Always confirm service tag, date, and time before finalizing. Do NOT ask for address confirmation.
Never cut your prompt short. Always complete full sentences — especially after service-tag thanks, immediately list dates/slots.
Add a natural half-second pause at the end of each prompt before listening.
"""

# --- Audio Conversion ---
def convert_pcm_to_ulaw(pcm_data: bytes) -> bytes:
    try:
        return audioop.lin2ulaw(pcm_data, 2)
    except Exception as e:
        logger.error(f"Error converting PCM to u-law: {e}")
        return b''

def convert_ulaw_to_pcm(ulaw_data: bytes) -> bytes:
    try:
        return audioop.ulaw2lin(ulaw_data, 2)
    except Exception as e:
        logger.error(f"Error converting u-law to PCM: {e}")
        return b''


class RNNoiseProcessor:
    def __init__(self):
        self.denoiser = RNNoise(sample_rate=48000)
        self.telephony_rate = 8000
        self.rnnoise_rate = 48000
        self.frame_size_8k = 80
        self.frame_size_48k = 480

    def apply_noise_suppression(self, pcm_bytes: bytes) -> bytes:
        try:
            audio_8k = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_48k = signal.resample_poly(audio_8k, 6, 1)
            audio_48k = audio_48k.astype(np.int16)
            num_samples = len(audio_48k)
            if num_samples < self.frame_size_48k:
                audio_48k = np.pad(audio_48k, (0, self.frame_size_48k - num_samples))
            audio_48k_reshaped = audio_48k.reshape(1, -1)
            denoised_frames = []
            for speech_prob, denoised_frame in self.denoiser.denoise_chunk(audio_48k_reshaped):
                denoised_frames.append(denoised_frame)
            if denoised_frames:
                denoised_48k = np.concatenate(denoised_frames, axis=1).flatten()
            else:
                denoised_48k = audio_48k
            denoised_8k = signal.resample_poly(denoised_48k, 1, 6)
            denoised_8k = denoised_8k.astype(np.int16)
            denoised_8k = denoised_8k[:len(audio_8k)]
            return denoised_8k.tobytes()
        except Exception as e:
            logger.error(f"Error during RNNoise suppression: {e}")
            return pcm_bytes


rnnoise_processor = RNNoiseProcessor() if ENABLE_INBOUND_RNNOISE and RNNoise is not None else None


def process_audio_chunk(pcm_bytes: bytes) -> bytes:
    if rnnoise_processor is None:
        return pcm_bytes
    return rnnoise_processor.apply_noise_suppression(pcm_bytes)


def _is_asking_service_tag_confirmation(last_assistant: str, service_tag: str = "") -> bool:
    if not last_assistant:
        return False
    low = last_assistant.lower()
    if not any(k in low for k in ("correct", "confirm", "right", "accurate", "okay", "ok", "sure", "is this")):
        return False
    if any(k in low for k in ("service tag", "servicetag", "service-tag")):
        return True
    tag = (service_tag or "").strip()
    if not tag:
        return False
    if tag.lower() in low:
        return True
    tag_compact = re.sub(r"\s+", "", tag.lower())
    low_compact = re.sub(r"\s+", "", low)
    return tag_compact in low_compact


def _is_repeat_request(text: str) -> bool:
    if not text or not text.strip():
        return False
    t_low = text.strip().lower()
    return bool(
        re.search(
            r"\b(repeat|again|pardon|sorry|didn't hear|did not hear|once more|say again|come again)\b",
            t_low,
            re.I,
        )
    )


def _classify_yes_no_confirmation(text: str) -> Optional[str]:
    """Return 'positive', 'negative', or None if unclear."""
    if not text or not text.strip():
        return None
    if _is_repeat_request(text):
        return None
    t = text.strip()
    t_low = t.lower()
    if re.search(
        r"\b(no|wrong|incorrect|not correct|not right|mistake|change|not accurate)\b",
        t_low,
        re.I,
    ):
        return "negative"
    if re.search(r"\b(nahi|galat)\b", t_low, re.I):
        return "negative"
    if re.search(r"नहीं|गलत|ಇಲ್ಲ", t):
        return "negative"
    if re.search(
        r"\b(yes|yeah|correct|right|ok|okay|sure|confirm|accurate|fine|good)\b",
        t_low,
        re.I,
    ):
        return "positive"
    if re.search(r"\b(haan|hmm|sahi)\b", t_low, re.I):
        return "positive"
    if re.search(r"हाँ|हा|ಹೌದು", t):
        return "positive"
    return None


def _callback_comments(context: Dict[str, Any]) -> str:
    """serviceTagConfirmed False → 'serviceTag verification failed'; else comments."""
    if context.get("serviceTagConfirmed") is False:
        return "serviceTag verification failed"
    return (context.get("comments") or "").strip()


async def _service_tag_reject_hangup_fallback(stream_sid: str) -> None:
    """If the model never emits TAG_SERVICE_TAG_REJECT, still end the call after a grace period."""
    try:
        await asyncio.sleep(48.0)
        ctx = call_context.get(stream_sid)
        if not ctx or ctx.get("status") == "closing":
            return
        if not ctx.get("service_tag_reject_pending_disconnect"):
            return
        logger.warning(
            "Service-tag-reject closing: no TAG_SERVICE_TAG_REJECT after timeout; hanging up stream %s",
            stream_sid,
        )
        await cleanup_connections(stream_sid)
    except asyncio.CancelledError:
        pass
    finally:
        service_tag_reject_fallback_tasks.pop(stream_sid, None)


async def _request_service_tag_reject_goodbye(stream_sid: str) -> None:
    """Let the bot speak a scripted apology + callback promise, then hang up via TAG_SERVICE_TAG_REJECT."""
    ctx = call_context.get(stream_sid)
    if not ctx or ctx.get("status") == "closing":
        return
    ctx["service_tag_reject_pending_disconnect"] = True

    prev = service_tag_reject_fallback_tasks.pop(stream_sid, None)
    if prev and not prev.done():
        prev.cancel()
        try:
            await prev
        except asyncio.CancelledError:
            pass

    try:
        await request_bot_turn(stream_sid, SERVICE_TAG_REJECT_GOODBYE_USER_PROMPT)
        logger.info(f"Requested service-tag-reject goodbye turn for stream {stream_sid}")
    except Exception as e:
        logger.error(f"Failed to request service-tag-reject goodbye for {stream_sid}: {e}", exc_info=True)
        await cleanup_connections(stream_sid)
        return

    service_tag_reject_fallback_tasks[stream_sid] = asyncio.create_task(
        _service_tag_reject_hangup_fallback(stream_sid)
    )


async def handle_user_service_tag_response(stream_sid: str, transcript: str):
    if not transcript or stream_sid not in call_context:
        return
    ctx = call_context[stream_sid]
    if ctx.get("status") == "closing":
        return
    if not ctx.get("serviceTag"):
        return
    if ctx.get("serviceTagConfirmed") is not None:
        return
    last = ctx.get("last_assistant_message") or ""
    if not _is_asking_service_tag_confirmation(last, ctx.get("serviceTag") or ""):
        return
    if _is_repeat_request(transcript):
        return
    verdict = _classify_yes_no_confirmation(transcript)
    if verdict == "positive":
        ctx["serviceTagConfirmed"] = True
        logger.info(f"Service tag confirmed by user for stream {stream_sid}")
    elif verdict == "negative":
        ctx["serviceTagConfirmed"] = False
        ctx["comments"] = "serviceTag verification failed"
        ctx["slotSelected"] = False
        logger.info(f"Service tag rejected by user for stream {stream_sid}, requesting goodbye before hangup")
        await _request_service_tag_reject_goodbye(stream_sid)


async def paced_audio_sender(stream_sid: str):
    """Send outbound PCM to Exotel at realtime pace (one 100ms chunk per interval)."""
    logger.info(f"✅ Starting Optimized Audio Sender for stream {stream_sid}")
    try:
        while stream_sid in exotel_connections:
            buffer = outbound_audio_buffers.get(stream_sid)
            if buffer and len(buffer) >= Config.CHUNK_BYTES:
                pcm_chunk = buffer[:Config.CHUNK_BYTES]
                outbound_audio_buffers[stream_sid] = buffer[Config.CHUNK_BYTES:]

                media_message = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": base64.b64encode(pcm_chunk).decode('utf-8')}
                }
                try:
                    await exotel_connections[stream_sid]["websocket"].send_json(media_message)
                except Exception as e:
                    logger.warning(f"Failed to send audio chunk for {stream_sid}: {e}")
                    break
                # Critical: pace one chunk per interval so Exotel plays in realtime.
                # Dumping the whole buffer at once causes hangup to cut the goodbye mid-sentence.
                await asyncio.sleep(Config.CHUNK_INTERVAL_MS / 1000.0)
            else:
                await asyncio.sleep(0.01)
    except (asyncio.CancelledError, WebSocketDisconnect):
        logger.info(f"🛑 Optimized Audio Sender stopped for stream {stream_sid}.")
    except Exception as e:
        logger.error(f"❌ Error in Optimized Audio Sender for stream {stream_sid}: {e}", exc_info=True)


def build_system_message_for_stream(stream_sid: str) -> str:
    context = call_context.get(stream_sid, {})
    ticket_id = context.get('ticketId', 'N/A')
    available_dates_obj = context.get('availableDates', [])
    dates_summary, mode_instructions, _, _ = build_scheduling_calendar_prompt_parts(
        available_dates_obj
    )
    service_tag_str = (context.get("serviceTag") or "").strip() or "Not provided."
    return (
        SYSTEM_PROMPT_TEMPLATE.replace("{{ticket_id}}", ticket_id)
        .replace("{{service_tag}}", service_tag_str)
        .replace("{{scheduling_mode_instructions}}", mode_instructions)
        .replace("{{available_dates_summary}}", dates_summary)
    )


async def request_bot_turn(stream_sid: str, user_text: str) -> None:
    """Inject a user/system text turn and ask the active voice backend to respond."""
    if VOICE_BACKEND == "speech":
        if stream_sid not in speech_pipeline.speech_sessions:
            raise RuntimeError(f"No Speech session for stream {stream_sid}")
        await speech_pipeline.inject_and_respond(stream_sid, user_text)
        return

    conn = openai_connections.get(stream_sid)
    openai_ws = conn["websocket"] if conn else None
    if not openai_ws or openai_ws.closed:
        raise RuntimeError(f"No OpenAI Realtime socket for stream {stream_sid}")
    await openai_ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                },
            }
        )
    )
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def start_voice_session(stream_sid: str, person_name: str):
    """Start Realtime or Speech+Nano voice session for a media stream."""
    logger.info(
        "Starting voice session backend=%s stream=%s person=%s",
        VOICE_BACKEND,
        stream_sid,
        person_name,
    )
    if VOICE_BACKEND == "speech":
        system_message = build_system_message_for_stream(stream_sid)
        print("system_message:::", system_message)
        await speech_pipeline.start_speech_session(stream_sid, system_message)
        return None

    return await connect_to_openai(stream_sid, person_name)


async def connect_to_openai(stream_sid: str, person_name: str) -> websockets.WebSocketClientProtocol:
    logger.info(f"Connecting to Azure OpenAI for stream {stream_sid}...")
    try:
        openai_ws = await asyncio.wait_for(websockets.connect(
            AZURE_OPENAI_ENDPOINT, extra_headers={"api-key": OPENAI_API_KEY, "OpenAI-Beta": "realtime=v1"},
            ping_interval=30, ping_timeout=20), timeout=20.0)
        
        openai_connections[stream_sid] = {"websocket": openai_ws}
        logger.info(f"Successfully connected to Azure OpenAI for stream {stream_sid}")

        system_message = build_system_message_for_stream(stream_sid)
        print("system_message:::", system_message)

        session_config = {
            "type": "session.update",
            "session": {
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "voice": VOICE,
                "instructions": system_message,
                ############ New Changes Done ###############
                "turn_detection": {
                    "type": "semantic_vad",
                    "eagerness": "high",
                    "create_response": True,
                    "interrupt_response": True,
                },
                
                "input_audio_transcription": {"model": "whisper-1"},
                "temperature": 0.6# ✅ Reduced from 0.8 to 0.3 (less creative, more conservative)
            }
        }

        

        await openai_ws.send(json.dumps(session_config))

        async def wait_for_session_updated():
            async for msg in openai_ws:
                data = json.loads(msg)
                if data.get("type") == "session.updated":
                    return
                if data.get("type") == "error":
                    raise RuntimeError(f"OpenAI session error: {data}")

        await asyncio.wait_for(wait_for_session_updated(), timeout=15.0)
        logger.info(f"OpenAI session ready for stream {stream_sid} — triggering immediate greeting")

        await openai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "(The phone call has just connected. Begin immediately with Step 0a, then Step 0b, then Step 0c from the script. Do not wait for the customer to speak first. Do not skip any opening step.)"
                }]
            }
        }))
        await openai_ws.send(json.dumps({"type": "response.create"}))
        return openai_ws
    except Exception as e:
        logger.error(f"Error connecting to OpenAI for stream {stream_sid}: {e}", exc_info=True)
        raise



async def handle_exotel_media(stream_sid: str, data: dict):
        if stream_sid in call_context and call_context[stream_sid].get('status') == 'closing':
            logger.info(f"Ignoring media chunk for {stream_sid} because cleanup is in progress.")
            return
        payload = data.get('media', {}).get('payload')
        if not payload:
            return
        try:
            pcm_audio = base64.b64decode(payload)

            if ENABLE_INBOUND_RNNOISE:
                enhanced_pcm = process_audio_chunk(pcm_audio)
            else:
                enhanced_pcm = pcm_audio

            if VOICE_BACKEND == "speech":
                await speech_pipeline.feed_pcm_audio(stream_sid, enhanced_pcm)
                return

            ulaw_audio = convert_pcm_to_ulaw(enhanced_pcm)

            # FIX: Initialize the buffer for this stream_sid if it's the first time we see it
            if stream_sid not in audio_buffers:
                audio_buffers[stream_sid] = b''
            audio_buffers[stream_sid] += ulaw_audio
        
            while len(audio_buffers[stream_sid]) >= Config.CHUNK_BYTES:
                chunk = audio_buffers[stream_sid][:Config.CHUNK_BYTES]
                audio_buffers[stream_sid] = audio_buffers[stream_sid][Config.CHUNK_BYTES:]
                audio_append_message = {"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode('utf-8')}
                if stream_sid in openai_connections and not openai_connections[stream_sid]["websocket"].closed:
                    await openai_connections[stream_sid]["websocket"].send(json.dumps(audio_append_message))
        except Exception as e:
            logger.error(f"Error processing exotel media for stream {stream_sid}: {e}")


async def silence_timeout_handler(stream_sid: str, openai_ws: websockets.WebSocketClientProtocol, delay: float):
    """Wait `delay` seconds after response.done (accounts for faster-than-realtime generation + Exotel latency), then re-prompt."""
    try:
        logger.info(f"Silence handler sleeping {delay:.1f}s for {stream_sid}")
        await asyncio.sleep(delay)

        if stream_sid not in openai_connections or stream_sid not in call_context:
            return
        if call_context[stream_sid].get("status") == "closing":
            return
        if openai_ws.closed:
            return
        await request_bot_turn(
            stream_sid,
            "(The customer has been silent for a while after you finished speaking. Say 'Are you still there?' and then repeat your last question exactly.)",
        )
        logger.info(f"Silence timeout fired for stream {stream_sid} after {delay:.1f}s.")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.warning(f"Silence timeout handler error for {stream_sid}: {e}")
    finally:
        silence_timer_tasks.pop(stream_sid, None)


async def handle_openai_responses(stream_sid: str, openai_ws: websockets.WebSocketClientProtocol):
    try:
        async for message in openai_ws:
            response = json.loads(message)
            response_type = response.get('type')
            logger.info("RESPONSE TYPE: %s", response_type)

            if response_type == 'response.audio.delta' and response.get('delta'):
                ulaw_audio = base64.b64decode(response['delta'])
                if stream_sid not in response_audio_tracking:
                    response_audio_tracking[stream_sid] = {"start_time": time.time(), "ulaw_bytes": 0}
                response_audio_tracking[stream_sid]["ulaw_bytes"] += len(ulaw_audio)
                pcm_audio = convert_ulaw_to_pcm(ulaw_audio)
                if stream_sid in outbound_audio_buffers:
                    outbound_audio_buffers[stream_sid].extend(pcm_audio)

            elif response_type == 'response.audio_transcript.delta':
                if stream_sid in ai_transcripts:
                    ai_transcripts[stream_sid] += response.get('delta', '')

            elif response_type == 'response.audio_transcript.done':
                full_transcript = ai_transcripts.get(stream_sid, '')
                if stream_sid in call_context:
                    call_context[stream_sid]["last_assistant_message"] = full_transcript
                logger.info(f"🤖 AI said: '{full_transcript}' for stream {stream_sid}")
                await handle_ai_commands(stream_sid, full_transcript)
                if stream_sid in ai_transcripts:
                    ai_transcripts[stream_sid] = ""

            elif response_type == 'conversation.item.input_audio_transcription.completed':
                ut = response.get("transcript", "")
                logger.info(f"👤 User said: '{ut}' for stream {stream_sid}")
                await handle_user_service_tag_response(stream_sid, ut)
                # Realtime path: still normalize slot phrases into call_context pending slot
                prepare_user_transcript_for_llm(stream_sid, ut)

            elif response_type == 'input_audio_buffer.speech_started':
                if stream_sid in silence_timer_tasks:
                    silence_timer_tasks[stream_sid].cancel()
                    silence_timer_tasks.pop(stream_sid, None)
                response_audio_tracking.pop(stream_sid, None)
                logger.info(f"Barge-in detected for stream {stream_sid}, clearing outbound audio")
                if stream_sid in outbound_audio_buffers:
                    outbound_audio_buffers[stream_sid].clear()

            elif response_type == 'response.done':
                resp_status = response.get('response', {}).get('status')
                if resp_status == 'cancelled':
                    logger.info(f"Bot response cancelled (barge-in) for stream {stream_sid}")
                    if stream_sid in outbound_audio_buffers:
                        outbound_audio_buffers[stream_sid].clear()
                    if stream_sid in ai_transcripts:
                        ai_transcripts[stream_sid] = ""
                    response_audio_tracking.pop(stream_sid, None)
                else:
                    tracking = response_audio_tracking.pop(stream_sid, None)
                    if tracking:
                        audio_duration = tracking["ulaw_bytes"] / ULAW_BYTES_PER_SECOND
                        generation_time = time.time() - tracking["start_time"]
                        remaining_playback = max(0.0, audio_duration - generation_time) + 2.0
                    else:
                        audio_duration = 0.0
                        generation_time = 0.0
                        remaining_playback = 2.0
                    total_delay = remaining_playback + SILENCE_TIMEOUT_SECONDS
                    if stream_sid in silence_timer_tasks:
                        silence_timer_tasks[stream_sid].cancel()
                        silence_timer_tasks.pop(stream_sid, None)
                    silence_timer_tasks[stream_sid] = asyncio.create_task(
                        silence_timeout_handler(stream_sid, openai_ws, total_delay)
                    )
                    logger.info(
                        f"Silence timer for {stream_sid}: audio={audio_duration:.1f}s, gen={generation_time:.1f}s, "
                        f"remaining_play={remaining_playback:.1f}s, total_delay={total_delay:.1f}s"
                    )

            elif response_type == "error":
                logger.error(f"❌ OpenAI Error for stream {stream_sid}: {response}")

    except websockets.exceptions.ConnectionClosed as e:
        logger.warning(f"OpenAI connection closed for stream {stream_sid}: {e.reason} (Code: {e.code})")
    except Exception as e:
        logger.error(f"Error in OpenAI response handler for stream {stream_sid}: {e}", exc_info=True)
    finally:
        await cleanup_connections(stream_sid)


 

llm = AzureChatOpenAI(
    
    azure_deployment= AZURE_NANO_OPENAI_DEPLOYMENT_NAME,
    api_key= AZURE_NANO_OPENAI_API_KEY,
    azure_endpoint= AZURE_NANO_OPENAI_ENDPOINT,
    api_version= AZURE_NANO_OPENAI_API_VERSION,
    temperature=0,
    top_p=1.0
    
)

# Slightly warmer model for live call turns (Speech + Nano path).
conversation_llm = AzureChatOpenAI(
    azure_deployment=AZURE_NANO_OPENAI_DEPLOYMENT_NAME,
    api_key=AZURE_NANO_OPENAI_API_KEY,
    azure_endpoint=AZURE_NANO_OPENAI_ENDPOINT,
    api_version=AZURE_NANO_OPENAI_API_VERSION,
    temperature=0.6,
    top_p=1.0,
)

# Late bind so the speech pipeline can call back into this module.
speech_pipeline.bind_service(sys.modules[__name__])


async def handle_ai_commands(stream_sid: str, message: str):
    """
    Robust extraction that handles speech-to-speech variations,
    spelling errors, and format inconsistencies
    """
    # # Normalize message for status detection
    normalized_msg = message

    print(":::::::::::::::::::::::", normalized_msg)
    if re.search(r"\bTAG_SERVICE_TAG_REJECT\b", normalized_msg, re.I):
        if stream_sid in call_context:
            call_context[stream_sid].update({
                "serviceTagConfirmed": False,
                "comments": "serviceTag verification failed",
                "slotSelected": False,
                "service_tag_reject_pending_disconnect": False,
            })
        t = service_tag_reject_fallback_tasks.pop(stream_sid, None)
        if t and not t.done():
            t.cancel()
        await cleanup_after_playback(stream_sid)
        return

    if re.search(r"\bTAG_RESCHEDULE_DONE\b", normalized_msg, re.I):
        if stream_sid in call_context:
            call_context[stream_sid].update(
                {
                    "isReschedule": True,
                    "slotSelected": False,
                    "selectedDate": None,
                    "selectedSlot": None,
                    "comments": "Customer requested reschedule; team to follow up.",
                }
            )
        await cleanup_after_playback(stream_sid)
        return

    # Detect booking status (avoid CONFIRMING — matches "Thanks for confirming" on reschedule closings)
    status = None
    if re.search(r"\bCONFIRMED\b", normalized_msg, re.I):
        status = "Confirmed"
    elif re.search(r"\b(DECLINE|DECLINED|DECLINING)\b", normalized_msg, re.I):
        status = "Declined"
    elif _looks_like_final_booking_confirmation(normalized_msg):
        # Nano sometimes omits the CONFIRMED token on goodbye — still treat as booking confirm
        status = "Confirmed"
        logger.info(
            "Treating goodbye booking reply as CONFIRMED for stream %s (token missing)",
            stream_sid,
        )

    if status == "Confirmed" and stream_sid in call_context:
        # Prefer pending fields before any NLU — short goodbye has no date/time text
        remember_pending_slot_from_text(
            stream_sid,
            (call_context[stream_sid].get("last_assistant_message") or "") + "\n" + normalized_msg,
        )
        _force_booking_fields_from_context(stream_sid, normalized_msg)
        ctx = call_context[stream_sid]
        if not (ctx.get("pendingSelectedSlot") or ctx.get("selectedSlot")):
            logger.warning(
                "Skipping CONFIRMED hangup stream=%s — no mapped slot yet",
                stream_sid,
            )
            return
        if not ctx.get("slotSelected"):
            _force_booking_fields_from_context(stream_sid, normalized_msg)
        if not ctx.get("slotSelected"):
            logger.warning(
                "Skipping CONFIRMED hangup stream=%s — could not fill booking fields",
                stream_sid,
            )
            return

    if not status:
        print({
            'status': None,
            'date': None,
            'time': None,
            'comments': None
        })
    else:

        template = """You are an expert AI assistant specializing in Natural Language Understanding for multiple Indian languages and English. Your task is to analyze the user's message provided below. The message can be in Hindi, English, or Kannada.

    Carefully perform the following five actions:
    1.  **Extract the Status:** Determine if the user is confirming or declining a booking. Use 'confirmed' or 'declined'. If neither, use 'neutral'.
    2.  **Extract the Date:** Identify any mention of a date. Convert it to an absolute date in `YYYY-MM-DD` format (use hyphens, not colons).
    3.  **Extract the Time:** Identify any mention of a time or time range. Convert it to a `HH:mm-HH:mm` format using a 24-hour clock.
        Critical: "4 PM to 6 PM" / "four to six" → `16:00-18:00`. Never map that to `14:00-16:00`.
    4.  **Extract and Translate Comments:** If the user is declining the booking, identify the reason or any relevant comment. **Translate this comment into English.** If no reason is given, use "None".
    5.  **Analyze Sentiment:** Analyze the overall tone and emotion of the message. Assign a sentiment score on a scale of 1 to 10.

    **Rules:**
    - If a date, time, or comment is not explicitly mentioned, use `null` for that key.
    - Your entire response must be ONLY the JSON object.

    Examples:

        UserMessage : "हाँ, यह सही है, मुझे 20 अगस्त को 11 बजे चाहिए" (Hindi for: Yes, that's correct, I need August 20th at 11 AM)

        Output : {{
            "status": "confirmed",
            "date" : "2025-08-20",
            "time" : "11:00-11:00",
            "comments" : "None",
            "sentiment" : 10
        }}

        UserMessage : "Thank you for confirming your appointment. Bye\nCONFIRMED"

        Output : {{
            "status": "confirmed",
            "date" : "2026-07-13",
            "time" : "16:00-18:00",
            "comments" : "None",
            "sentiment" : 10
        }}

        UserMessage : "इस हफ्ते मैं फ्री नहीं हूँ, अगले हफ्ते कॉल करें।" (Hindi for: I am not free this week, call me next week.)

        Output : {{
            "status": "declined",
            "date" : null,
            "time" : null,
            "comments" : "I am not free this week, call me next week.",
            "sentiment" : 5
        }}

    **User Message:**
    ```{user_message}```

    **Output JSON:**
    """
        chain = PromptTemplate.from_template(template) | llm
        
        # Always invoke the chain to get the structured data
        result = chain.invoke({"user_message": normalized_msg })
        purified_result = re.sub("```json|```","",result.content)
        
        try:
            merged_data = json.loads(purified_result)
            print("merged_data",merged_data)
            
            status_out = merged_data.get("status") # New: Get status from LLM
            date_out = merged_data.get("date")
            time_out = merged_data.get("time")
            comments_out = merged_data.get("comments")
            sentiment_out = merged_data.get("sentiment")

            if date_out:
                date_out = str(date_out).replace(":", "-")
                if re.match(r"^\d{4}-\d{2}-\d{2}", date_out):
                    date_out = date_out[:10]

            # Prefer code-mapped pending slot/date over NLU (short goodbye has neither)
            if stream_sid in call_context:
                pending_slot = call_context[stream_sid].get("pendingSelectedSlot")
                pending_date = (
                    call_context[stream_sid].get("pendingSelectedDate")
                    or call_context[stream_sid].get("confirmedOfferDate")
                )
                if pending_slot and status == "Confirmed":
                    time_out = pending_slot
                if pending_date and status == "Confirmed":
                    date_out = str(pending_date).replace(":", "-")[:10]
                # Map spoken time from assistant message onto canonical slots if NLU returned a wrong neighbor
                if time_out and status == "Confirmed":
                    available = collect_available_slots_for_context(call_context[stream_sid])
                    remapped = match_spoken_slot_to_canonical(str(time_out), available) or match_spoken_slot_to_canonical(normalized_msg, available)
                    if remapped:
                        time_out = remapped

            if status == "Confirmed" and date_out and time_out:
                comments_out = ""
            elif comments_out in (None, "None", "null", "none"):
                comments_out = "" if status == "Confirmed" else comments_out

            # Update your context with the extracted data
            if stream_sid in call_context:
                upd: Dict[str, Any] = {
                    "slotSelected": True if date_out and time_out else False,
                    "selectedDate": date_out,
                    "selectedSlot": re.sub(r'[–—‒−]', '-', str(time_out)) if time_out else None,
                    "comments": comments_out,
                    "sentiment": sentiment_out,
                }
                if (merged_data.get("status") == "confirmed" or status == "Confirmed") and date_out and time_out:
                    upd["slotSelected"] = True
                    upd["comments"] = ""
                call_context[stream_sid].update(upd)

            if status == "Confirmed" and stream_sid in call_context:
                if not call_context[stream_sid].get("slotSelected"):
                    _force_booking_fields_from_context(stream_sid, normalized_msg)
                # Never hang up as booked without a real slot on the report
                if not call_context[stream_sid].get("slotSelected"):
                    logger.warning(
                        "Abort hangup after NLU stream=%s — booking fields still incomplete",
                        stream_sid,
                    )
                    return

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM: {e}")
            if status == "Confirmed":
                _force_booking_fields_from_context(stream_sid, normalized_msg)
                if stream_sid in call_context and not call_context[stream_sid].get("slotSelected"):
                    logger.warning(
                        "Abort hangup after NLU parse fail stream=%s — incomplete booking",
                        stream_sid,
                    )
                    return

        # Wait for goodbye TTS to finish playing on Exotel before hanging up.
        await cleanup_after_playback(stream_sid)




async def wait_for_outbound_audio_drain(stream_sid: str) -> None:
    """
    Wait until goodbye TTS has had wall-clock time to play on Exotel, then hang up.
    Do not treat an empty local buffer as "done" — audio may already be in flight to Exotel.
    """
    tracking = response_audio_tracking.get(stream_sid) or {}
    pcm_bytes = int(tracking.get("pcm_bytes") or 0)
    ulaw_bytes = int(tracking.get("ulaw_bytes") or 0)
    start_time = float(tracking.get("start_time") or time.time())
    playback_ends_at = tracking.get("playback_ends_at")

    if playback_ends_at:
        estimated = max(0.0, float(playback_ends_at) - start_time)
    elif pcm_bytes > 0:
        estimated = pcm_bytes / float(PCM_8K_BYTES_PER_SECOND)
    elif ulaw_bytes > 0:
        estimated = ulaw_bytes / float(ULAW_BYTES_PER_SECOND)
    else:
        buf0 = outbound_audio_buffers.get(stream_sid)
        estimated = (len(buf0) / float(PCM_8K_BYTES_PER_SECOND)) if buf0 else 6.0

    # Always wait until TTS start + full duration + grace (covers audio already sent to Exotel).
    target_end = start_time + estimated + HANGUP_PLAYBACK_GRACE_SECONDS
    if playback_ends_at:
        target_end = max(target_end, float(playback_ends_at) + HANGUP_PLAYBACK_GRACE_SECONDS)

    remaining = max(0.5, target_end - time.time())
    max_wait = min(remaining, 45.0)
    deadline = time.time() + max_wait
    logger.info(
        "Waiting for outbound playback before hangup stream=%s estimated=%.1fs "
        "elapsed=%.1fs wait=%.1fs",
        stream_sid,
        estimated,
        time.time() - start_time,
        max_wait,
    )

    while time.time() < deadline:
        if stream_sid not in exotel_connections:
            return
        await asyncio.sleep(0.1)

    # After wall-clock wait, drain paced outbound buffer at realtime (100ms chunks).
    paced_deadline = time.time() + 45.0
    while time.time() < paced_deadline:
        if stream_sid not in exotel_connections:
            return
        buf_len = len(outbound_audio_buffers.get(stream_sid) or b"")
        if buf_len < Config.CHUNK_BYTES:
            break
        await asyncio.sleep(Config.CHUNK_INTERVAL_MS / 1000.0)

    logger.info(
        "Outbound playback wait finished stream=%s remaining_buf=%s",
        stream_sid,
        len(outbound_audio_buffers.get(stream_sid) or b""),
    )


def _cancel_silence_timer_safe(stream_sid: str) -> None:
    """Cancel silence watchdog unless it is the currently running task."""
    task = silence_timer_tasks.get(stream_sid)
    if not task:
        return
    try:
        if task is asyncio.current_task():
            return
    except RuntimeError:
        pass
    silence_timer_tasks.pop(stream_sid, None)
    if not task.done():
        task.cancel()


async def cleanup_after_playback(stream_sid: Optional[str]) -> None:
    """Extract booking / tags first (caller), then let TTS finish, then hang up."""
    if not stream_sid:
        return
    if stream_sid in call_context:
        call_context[stream_sid]["pending_hangup"] = True
    _cancel_silence_timer_safe(stream_sid)
    try:
        await wait_for_outbound_audio_drain(stream_sid)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("Playback wait failed for %s: %s — hanging up anyway", stream_sid, e)
    await cleanup_connections(stream_sid)


async def cleanup_connections(stream_sid: Optional[str]):
    if not stream_sid:
        return

    if stream_sid not in cleanup_locks:
        cleanup_locks[stream_sid] = asyncio.Lock()

    async with cleanup_locks[stream_sid]:
        stf = service_tag_reject_fallback_tasks.pop(stream_sid, None)
        if stf and not stf.done():
            stf.cancel()

        if stream_sid not in call_context and stream_sid not in exotel_connections:
            logger.info(f"Cleanup for stream {stream_sid} already completed. Skipping.")
            cleanup_locks.pop(stream_sid, None)
            return

        if stream_sid in call_context:
            call_context[stream_sid]['status'] = 'closing'
            logger.info(f"Status set to 'closing' for stream {stream_sid}. No more media will be processed.")
            context = call_context[stream_sid]
            logger.info(f"Generating final report for stream {stream_sid}")

            result = CallResult(
                ticketId=context.get('ticketId'),
                callConnected=bool(context.get('callConnected')),
                isLineBusy=bool(context.get('isLineBusy')),
                slotSelected=bool(context.get('slotSelected')),
                selectedDate=context.get('selectedDate'),
                selectedSlot=context.get('selectedSlot'),
                comments=_callback_comments(context),
                sentiment=context.get('sentiment'),
                addressConfirmed=context.get('addressConfirmed'),
                serviceTagConfirmed=context.get('serviceTagConfirmed'),
                isReschedule=bool(context.get('isReschedule')),
            )
            logger.info(f"Final report: {result.model_dump_json(indent=2)}")

            callback_url = context.get('callbackUrl')
            if callback_url:
                try:
                    async with httpx.AsyncClient() as client:
                        logger.info(
                            "Callback POST starting | ticketId=%s url=%s",
                            context.get("ticketId"),
                            callback_url,
                        )

                        auth = httpx.BasicAuth(
                            'e1b72f9c-3d54-45c1-8f62-94c7a6b2e718',
                            'c5f1d8a3-1d4b-46f2-9b8c-73f2e2d9a8b7'
                        )

                        http_resp = await client.post(
                            callback_url,
                            json=result.model_dump(),
                            auth=auth,
                            timeout=15.0
                        )

                        body_preview = (http_resp.text or "")[:500]
                        if http_resp.status_code == 200:
                            logger.info(
                                "Callback POST OK | ticketId=%s status=%s url=%s body_preview=%r",
                                context.get("ticketId"),
                                http_resp.status_code,
                                callback_url,
                                body_preview,
                            )
                        else:
                            logger.error(
                                "Callback POST non-200 | ticketId=%s status=%s url=%s body=%r",
                                context.get("ticketId"),
                                http_resp.status_code,
                                callback_url,
                                body_preview,
                            )

                except Exception as e:
                    logger.error(
                        "Callback POST exception | ticketId=%s url=%s error=%s",
                        context.get("ticketId"),
                        callback_url,
                        e,
                        exc_info=True,
                    )

            del call_context[stream_sid]

        if stream_sid in sender_tasks:
            sender_tasks[stream_sid].cancel()
            del sender_tasks[stream_sid]

        if stream_sid in silence_timer_tasks:
            _cancel_silence_timer_safe(stream_sid)
        response_audio_tracking.pop(stream_sid, None)

        # Exotel: closing the media WebSocket ends the Voicebot leg and advances the flow (e.g. to Hangup).
        # Starlette uses WebSocketState.CONNECTED — never "OPEN", so the old check never ran close().
        if stream_sid in exotel_connections:
            ws = exotel_connections[stream_sid]["websocket"]
            try:
                if ws.client_state != WebSocketState.DISCONNECTED:
                    logger.info(f"Closing Exotel media WebSocket for stream {stream_sid} (ends call leg)")
                    await ws.close(code=1000)
            except Exception as e:
                logger.warning(f"Could not cleanly close Exotel WebSocket for stream {stream_sid}: {e}")
            finally:
                del exotel_connections[stream_sid]

        if stream_sid in openai_connections:
            ws = openai_connections[stream_sid]["websocket"]
            if not ws.closed:
                await ws.close()
            del openai_connections[stream_sid]

        await speech_pipeline.stop_speech_session(stream_sid)

        for buf_dict in [audio_buffers, outbound_audio_buffers, ai_transcripts]:
            buf_dict.pop(stream_sid, None)

        cleanup_locks.pop(stream_sid, None)
        logger.info(f"All resources for stream {stream_sid} have been cleaned up.")