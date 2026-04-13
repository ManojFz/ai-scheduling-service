import re
import time
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
from pyrnnoise import RNNoise
from scipy import signal
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

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
if ENABLE_INBOUND_RNNOISE:
    logger.info("ENABLE_INBOUND_RNNOISE is enabled — inbound audio uses RNNoise")


# Azure OpenAI configuration

AZURE_OPENAI_ENDPOINT = "wss://fieldezai.cognitiveservices.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=gpt-realtime"
OPENAI_API_KEY = os.getenv("AZURE_NANO_OPENAI_API_KEY")



# gpt-4.1-nano
AZURE_NANO_OPENAI_ENDPOINT= os.getenv("AZURE_NANO_OPENAI_ENDPOINT")
AZURE_NANO_OPENAI_API_KEY= os.getenv("AZURE_NANO_OPENAI_API_KEY")
AZURE_NANO_OPENAI_DEPLOYMENT_NAME= os.getenv("AZURE_NANO_OPENAI_DEPLOYMENT_NAME")
AZURE_NANO_OPENAI_API_VERSION= os.getenv("AZURE_NANO_OPENAI_API_VERSION")



# --- Ensure mandatory keys are present ---
if not all([EXOTEL_API_KEY, EXOTEL_API_TOKEN, EXOTEL_SID, EXOTEL_FLOW_APP_ID, EXOTEL_CALLER_ID, OPENAI_API_KEY]):
    raise RuntimeError("Please set all required EXOTEL and AZURE_OPENAI_API_KEY environment variables!")

class TimeSlot(BaseModel):
    date: str
    slots: List[str]

class ScheduleCallRequest(BaseModel):
    ticketId: str
    customerPhone: str
    callbackUrl: HttpUrl # Use HttpUrl for validation
    address: str
    availableDates: List[TimeSlot]

# Model for the final call report
class CallResult(BaseModel):
    ticketId: str
    callConnected: bool = False
    slotSelected: bool = False
    selectedDate: Optional[str] = None
    selectedSlot: Optional[str] = None
    comments: Optional[str] = ""
    sentiment: Optional[int] = None
    addressConfirmed: Optional[bool] = None
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
address_reject_fallback_tasks: Dict[str, asyncio.Task] = {}
SILENCE_TIMEOUT_SECONDS = 6.0


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
        "address": "",
        "availableDates": [],
        "addressConfirmed": None,
        "last_assistant_message": "",
        "isReschedule": False,
    }


# Injected when user rejects address via STT; model must speak goodbye then end with TAG_ADDRESS_REJECT.
ADDRESS_REJECT_GOODBYE_USER_PROMPT = (
    "(System notice — the customer has indicated the service address is NOT correct. "
    "Speak ONE short closing only, in the customer's already-selected language (English, Hindi, or Kannada — "
    "same as the rest of this call). "
    "Include: apologize for the inconvenience; say you cannot continue scheduling without the correct address; "
    "say our team will get back to them soon (English example: "
    "\"Sorry for the inconvenience. We cannot continue without the correct address. "
    "We'll get back to you soon. Goodbye.\" — translate naturally for Hindi or Kannada); end with a brief goodbye. "
    "Do not ask any questions. Do not mention dates or time slots. "
    "On the VERY LAST LINE of your reply only, output exactly TAG_ADDRESS_REJECT (system use only; never speak it aloud)."
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


def _iter_available_date_rows(available_dates_obj: Any) -> List[tuple[str, List[str]]]:
    """Normalize payload `availableDates` to (YYYY-MM-DD, [slot strings])."""
    rows: List[tuple[str, List[str]]] = []
    for item in available_dates_obj or []:
        if isinstance(item, TimeSlot):
            rows.append((str(item.date).strip(), [str(s) for s in (item.slots or [])]))
        elif isinstance(item, dict):
            slots = item.get("slots") or []
            rows.append((str(item.get("date", "")).strip(), [str(s) for s in slots]))
        elif hasattr(item, "date") and hasattr(item, "slots"):
            rows.append(
                (
                    str(getattr(item, "date")).strip(),
                    [str(s) for s in list(getattr(item, "slots") or [])],
                )
            )
    return [r for r in rows if r[0]]


def build_scheduling_calendar_prompt_parts(available_dates_obj: Any) -> tuple[str, str, str, str]:
    """
    Build prompt injections for multi-date scheduling.
    Returns (available_dates_summary, scheduling_mode_instructions, scheduled_date_fallback, available_slots_fallback).
    """
    rows = _iter_available_date_rows(available_dates_obj)
    if not rows:
        summary = "  (No appointment dates were provided in the system payload.)"
        mode = (
            "NO_DATES_MODE: No valid dates were provided. After address confirmation, apologize briefly that no slots "
            "are available in the system and say our team will reach out; do not invent dates or times."
        )
        return summary, mode, "No date", ""

    lines: List[str] = []
    for d, slots in rows:
        slot_part = _slot_list_for_prompt(slots)
        lines.append(f"  - {d} (canonical YYYY-MM-DD — match customer speech to this row only): slots {slot_part}")
    summary = "\n".join(lines)

    if len(rows) == 1:
        d0, slots0 = rows[0]
        slot_list = _slot_list_for_prompt(slots0) if slots0 else ""
        mode = (
            "SINGLE_DATE_MODE: There is exactly ONE appointment date in the system list. After you thank the customer for "
            "confirming the address, do NOT ask them to choose among several dates. Go directly to time-slot selection "
            f"for date {d0} only. The only valid slots for that date are: {slot_list}. "
            "Speak each slot using the '(say aloud: …)' twelve-hour wording; never read raw hour numbers like '11 to 14'."
        )
        return summary, mode, d0, slot_list

    mode = (
        "MULTIPLE_DATE_MODE: After you thank the customer for confirming the address, ask them to choose an appointment DATE. "
        "List **every** date from the canonical list above in spoken form (full date in their language). "
        "Immediately after listing those dates, also offer a **reschedule option** (English example: "
        "\"Or if none of these dates work for you, do you need us to reschedule?\" — translate naturally for Hindi or Kannada). "
        "Do NOT read time slots until either (A) a listed date is chosen and confirmed, or (B) the customer chooses reschedule.\n"
        "PATH A — Pick a listed date: When they indicate one of the system dates, map to exactly one YYYY-MM-DD row. "
        "Confirm (English): \"So you would like [spoken date] — is that correct?\" Only after clear YES, present ONLY that date's time slots, "
        "then slot confirmation and booking flow as usual.\n"
        "PATH B — Reschedule: If they say they need to reschedule / none of these work / similar, first confirm intent "
        "(English example: \"You would like us to reschedule — is that correct?\"). "
        "Only after they clearly say YES: thank them for confirming, say you will note the reschedule request, "
        "that the team will get back to them soon, and goodbye. Do NOT book a slot. "
        "On the VERY LAST LINE only, output exactly TAG_RESCHEDULE_DONE (system use; never speak it aloud)."
    )
    first_d, first_slots = rows[0]
    return summary, mode, first_d, ", ".join(first_slots)


###################################### 2025 -09 -10 evening prompt ###############################################
SYSTEM_PROMPT_TEMPLATE = """
CORE DIRECTIVES
LANGUAGE SELECTION:
At the start of the conversation, you MUST first ask the customer to select their preferred language from: English, Hindi, or Kannada.
If the input is unclear, background noise, or not one of the supported languages, DO NOT auto-select.
Politely ask the customer to repeat:
👉 “I’m sorry, I didn’t catch that. Could you please say English, Hindi, or Kannada?”
Once a supported language is detected, confirm with the customer **in that detected language**:
👉 “You selected [Language]. Is that correct? ”
Only if the customer clearly says YES (or equivalent affirmation), lock in the language and proceed.
If the customer says NO or does not confirm, repeat the selection step again.
Do not proceed to the address step until the language is both detected AND confirmed.

STRICTLY FORBIDDEN:
NO MIXED LANGUAGES: Under NO circumstances are you to use any other language after the customer has confirmed their language preference. This rule applies to all prompts and all dynamic data. For example, if Hindi is selected, dates and times MUST be spoken only in Hindi, not a mix of Hindi and English.
NO INTERNAL DATA: Never speak or reference internal system commands, JSON data, sentiment scores, booking commands, numbers used for internal purposes, hangup commands, or tokens such as TAG_ADDRESS_REJECT or TAG_RESCHEDULE_DONE.
DO NOT READ BRACKETS: Never speak or read aloud any text inside curly braces {} or square brackets []. These are internal system placeholders or instructions, not part of the script to be spoken to the customer.

PERSONA:
You are a friendly and efficient AI scheduling assistant for FieldEZ. Your tone should be clear, friendly, and natural, not robotic.

STRICT SELECTION RULE:
Every valid appointment date and time range comes **only** from the canonical system list below. You must not invent, change, or merge dates or slots. Do not choose a date or time slot on behalf of the customer. If you are unsure what they chose, ask again. Do not assume.

TIME SLOT SPEECH (mandatory):
Payloads may show windows like `11:00-14:00` (24-hour). You MUST speak them in **twelve-hour AM/PM** using the `(say aloud: …)` cue next to each slot in the system list (e.g. **11 AM to 2 PM**, not "11 to 14"). **Never** read only the hour digits as two numbers (e.g. never "nine to eleven" for `09:00-11:00` in the wrong style, and never "eleven to fourteen"). In Hindi/Kannada, express the same twelve-hour meaning naturally.

BACKGROUND NOISE HANDLING:
If the response is unclear, garbled, or nonsensical, DO NOT guess. Politely ask them to repeat in their selected language.

YOUR TASK
Ticket ID for this call: {{ticket_id}}
The customer’s service address (use exactly this text where the script says to read the address): {{customer_address}}

{{scheduling_mode_instructions}}

Canonical schedule from the system (never offer dates or slots that are not listed here):
{{available_dates_summary}}

MANDATORY CONVERSATION FLOW
Step 0 (LANGUAGE SELECTION)
👉 “Hello! I'm calling from DELL Service to schedule your appointment. To better assist you, please select your preferred language: English, Hindi, or Kannada.”
If unclear/noise/invalid → repeat request.
Once detected → MUST confirm **in that detected language**:
👉 “You selected [Language]. Is that correct?”
Only proceed after YES/affirmation.

Step 1 (SERVICE ADDRESS CONFIRMATION)
Immediately after language is locked in, speak **in the customer’s selected language** using this structure (for English, follow it closely; for other languages, translate the same meaning naturally):
👉 “Your appointment is scheduled at the following address:
{{customer_address}}.
Please confirm if this address is correct.”
Read the address line exactly as given above ({{customer_address}}). Do not change spelling or omit parts of the address when you read it aloud.
Do **not** ask the customer to provide, spell, or dictate a different or “correct” address. The only question is whether **this** address is correct (yes/no, correct/wrong, or equivalent).
Rules:
If the customer says NO, wrong, incorrect, not correct, or clearly rejects the address → apologize briefly (e.g. sorry for the inconvenience), say you cannot continue without the correct address, say our team will get back to them soon, say goodbye, and **end the conversation**. Do NOT ask for dates or times.
If you must disconnect for a wrong address, put the exact token TAG_ADDRESS_REJECT alone on the very last line of your response (system use only; do not speak this token aloud).
If you receive a system notice that the address was rejected, follow it exactly: speak that closing apology and callback promise in the customer's language, then TAG_ADDRESS_REJECT on the last line only.
If the customer says YES, correct, right, or clearly confirms the address → **first** acknowledge with a short thanks for confirming the address **in their selected language** (English example: “Thanks for confirming the address.”), **then** follow **{{scheduling_mode_instructions}}** for date and slot steps — do not skip this thanks.
If unclear or noise → ask them to repeat. Do not assume.

Step 2 (DATE AND TIME — obey {{scheduling_mode_instructions}})

SINGLE_DATE_MODE:
After thanks for confirming the address, go straight to time slots for the **only** date in the canonical list. In the customer’s language (English example; translate for Hindi or Kannada):
👉 “We can schedule your appointment. For [spoken form of that date], the available time slots are: [list **only** valid slots for that date in 12-hour AM/PM]. Please choose one.”
If unclear → ask them to repeat. Do not auto-select a slot. After they choose, confirm: “You selected [Time Slot]. Is that correct?” Only after YES → continue toward Step 4.

MULTIPLE_DATE_MODE:
2a — After thanks for confirming the address, ask them to **select an appointment date**, list **only** the dates from the canonical list (spoken in full form), **then** add one more option: if they need **reschedule** instead because none of those dates work (see {{scheduling_mode_instructions}}). Do not read time slots yet.
2b — **Pick a listed date:** match their choice to one YYYY-MM-DD row. Confirm: “So you would like [spoken date] — is that correct?” Only after clear YES continue.
2c — Offer **only** the slots for that confirmed date. Confirm slot with YES before proceeding.
2d — **Reschedule path:** If they want reschedule, confirm (English): “You would like us to reschedule — is that correct?” After clear YES: thank them, say you will reschedule and the team will get back to them soon, goodbye. Last line only: TAG_RESCHEDULE_DONE (never spoken). End the call; do not offer slots.

NO_DATES_MODE:
After address confirmation, explain politely that no schedule was loaded and our team will follow up; do not invent slots.

Shared: Never offer slots for a date the customer has not confirmed in MULTIPLE_DATE_MODE.

Step 3 (BOOKING CONFIRMATION - INTERNAL CHECK)
(Internal Check - Do Not Speak)
You MUST NOT proceed to Step 4 until:
1. Confirmed service address from Step 1.
2. MULTIPLE_DATE_MODE: customer has **confirmed with YES** which appointment date (one YYYY-MM-DD from the list) before slots were offered for that date.
3. SINGLE_DATE_MODE: the single system date is the appointment date for slot selection.
4. A clear **YES** for the chosen **time slot** after you asked “Is that correct?” for the slot.

Formatting Rule for Step 4:
[Spoken Date] = the appointment date being booked (the only date in SINGLE_DATE_MODE, or the date the customer confirmed in MULTIPLE_DATE_MODE). Say it as ‘Month Day, Year’ in their language.
[Spoken Time] = chosen slot in 12-hour AM/PM (e.g., “9 AM to 11 AM”) in their language.
Raw codes and internal JSON are for system use only and must not be read aloud.

Action: When all conditions above are met, proceed immediately to Step 4.

Step 4 (FINAL TURN)
✅ If the customer CONFIRMS the booking, strictly respond (in their selected language). You must say the date and time in the spoken format as defined in Step 3.
English:
"Thank you for booking with us. Your appointment is CONFIRMED for [Spoken Date] at [Spoken Time]. Looking forward to helping you! Good Bye..."
Hindi:
"हमारे साथ बुकिंग करने के लिए धन्यवाद। आपकी अपॉइंटमेंट [Spoken Date] को [Spoken Time] के लिए CONFIRMED की गई है। आपको मदद करने की प्रतीक्षा रहेगी! अलविदा..."
Kannada:
"ನಮ್ಮೊಂದಿಗೆ ಬುಕ್ಕಿಂಗ್ ಮಾಡಿದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು. ನಿಮ್ಮ ಅಪಾಯಿಂಟ್ಮೆಂಟ್ [Spoken Date] ರಂದು [Spoken Time]ಕ್ಕೆ CONFIRMED ಮಾಡಲಾಗಿದೆ. ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ನಾವು ಎದುರು ನೋಡುತ್ತಿದ್ದೇವೆ! ವಿದಾಯ..."

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
Always confirm language, service address, date, and time before finalizing.
Never cut your prompt short. Always complete full sentences.
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


rnnoise_processor = RNNoiseProcessor()


def process_audio_chunk(pcm_bytes: bytes) -> bytes:
    return rnnoise_processor.apply_noise_suppression(pcm_bytes)


def _is_asking_address_confirmation(last_assistant: str) -> bool:
    if not last_assistant:
        return False
    low = last_assistant.lower()
    if not any(k in low for k in ("address", "location", "visit")):
        return False
    return any(k in low for k in ("correct", "confirm", "right", "accurate", "okay", "ok", "sure"))


def _classify_address_confirmation(text: str) -> Optional[str]:
    """Return 'positive', 'negative', or None if unclear."""
    if not text or not text.strip():
        return None
    t = text.strip()
    t_low = t.lower()
    if re.search(
        r"\b(no|wrong|incorrect|not correct|not right|mistake|change|not accurate|bad address|different address)\b",
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
    """addressConfirmed False → 'Address is wrong'; True → ''."""
    ac = context.get("addressConfirmed")
    if ac is False:
        return "Address is wrong"
    if ac is True:
        return ""
    return (context.get("comments") or "").strip()


async def _address_reject_hangup_fallback(stream_sid: str) -> None:
    """If the model never emits TAG_ADDRESS_REJECT, still end the call after a grace period."""
    try:
        await asyncio.sleep(48.0)
        ctx = call_context.get(stream_sid)
        if not ctx or ctx.get("status") == "closing":
            return
        if not ctx.get("address_reject_pending_disconnect"):
            return
        logger.warning(
            "Address-reject closing: no TAG_ADDRESS_REJECT after timeout; hanging up stream %s",
            stream_sid,
        )
        await cleanup_connections(stream_sid)
    except asyncio.CancelledError:
        pass
    finally:
        address_reject_fallback_tasks.pop(stream_sid, None)


async def _request_address_reject_goodbye(stream_sid: str) -> None:
    """Let the bot speak a scripted apology + callback promise, then hang up via TAG_ADDRESS_REJECT."""
    ctx = call_context.get(stream_sid)
    if not ctx or ctx.get("status") == "closing":
        return
    ctx["address_reject_pending_disconnect"] = True

    prev = address_reject_fallback_tasks.pop(stream_sid, None)
    if prev and not prev.done():
        prev.cancel()
        try:
            await prev
        except asyncio.CancelledError:
            pass

    conn = openai_connections.get(stream_sid)
    openai_ws = conn["websocket"] if conn else None
    if not openai_ws or openai_ws.closed:
        logger.warning("No OpenAI socket for address-reject goodbye; hanging up stream %s", stream_sid)
        await cleanup_connections(stream_sid)
        return

    try:
        await openai_ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": ADDRESS_REJECT_GOODBYE_USER_PROMPT}],
                    },
                }
            )
        )
        await openai_ws.send(json.dumps({"type": "response.create"}))
        logger.info(f"Requested address-reject goodbye turn for stream {stream_sid}")
    except Exception as e:
        logger.error(f"Failed to request address-reject goodbye for {stream_sid}: {e}", exc_info=True)
        await cleanup_connections(stream_sid)
        return

    address_reject_fallback_tasks[stream_sid] = asyncio.create_task(
        _address_reject_hangup_fallback(stream_sid)
    )


async def handle_user_address_response(stream_sid: str, transcript: str):
    if not transcript or stream_sid not in call_context:
        return
    ctx = call_context[stream_sid]
    if ctx.get("status") == "closing":
        return
    if not ctx.get("address"):
        return
    if ctx.get("addressConfirmed") is not None:
        return
    last = ctx.get("last_assistant_message") or ""
    if not _is_asking_address_confirmation(last):
        return
    verdict = _classify_address_confirmation(transcript)
    if verdict == "positive":
        ctx["addressConfirmed"] = True
        ctx["comments"] = ""
        logger.info(f"Address confirmed by user for stream {stream_sid}")
    elif verdict == "negative":
        ctx["addressConfirmed"] = False
        ctx["comments"] = "Address is wrong"
        ctx["slotSelected"] = False
        logger.info(f"Address rejected by user for stream {stream_sid}, requesting goodbye before hangup")
        await _request_address_reject_goodbye(stream_sid)


async def paced_audio_sender(stream_sid: str):
    logger.info(f"✅ Starting Optimized Audio Sender for stream {stream_sid}")
    prebuffer_threshold = Config.CHUNK_BYTES * 2
    try:
        while stream_sid in exotel_connections:
            buffer = outbound_audio_buffers.get(stream_sid)
            if buffer and len(buffer) >= Config.CHUNK_BYTES:
                while len(buffer) >= Config.CHUNK_BYTES:
                    pcm_chunk = buffer[:Config.CHUNK_BYTES]
                    outbound_audio_buffers[stream_sid] = buffer[Config.CHUNK_BYTES:]
                    buffer = outbound_audio_buffers[stream_sid]
                    
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
            
            buffer_len = len(outbound_audio_buffers.get(stream_sid, b''))
            if buffer_len < prebuffer_threshold:
                await asyncio.sleep(0.005)
            else:
                await asyncio.sleep(Config.CHUNK_INTERVAL_MS / 1000.0)
    except (asyncio.CancelledError, WebSocketDisconnect):
        logger.info(f"🛑 Optimized Audio Sender stopped for stream {stream_sid}.")
    except Exception as e:
        logger.error(f"❌ Error in Optimized Audio Sender for stream {stream_sid}: {e}", exc_info=True)


async def connect_to_openai(stream_sid: str, person_name: str) -> websockets.WebSocketClientProtocol:
    logger.info(f"Connecting to Azure OpenAI for stream {stream_sid}...")
    try:
        openai_ws = await asyncio.wait_for(websockets.connect(
            AZURE_OPENAI_ENDPOINT, extra_headers={"api-key": OPENAI_API_KEY, "OpenAI-Beta": "realtime=v1"},
            ping_interval=30, ping_timeout=20), timeout=20.0)
        
        openai_connections[stream_sid] = {"websocket": openai_ws}
        logger.info(f"Successfully connected to Azure OpenAI for stream {stream_sid}")

        context = call_context.get(stream_sid, {})
        ticket_id = context.get('ticketId', 'N/A')
        available_dates_obj = context.get('availableDates', [])
        dates_summary, mode_instructions, _, _ = build_scheduling_calendar_prompt_parts(
            available_dates_obj
        )

        address_str = context.get("address") or "Not provided."
        system_message = (
            SYSTEM_PROMPT_TEMPLATE.replace("{{ticket_id}}", ticket_id)
            .replace("{{scheduling_mode_instructions}}", mode_instructions)
            .replace("{{available_dates_summary}}", dates_summary)
            .replace("{{customer_address}}", address_str)
        )
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
                    "text": "(The phone call has just connected. Begin immediately with your opening greeting from the script. Do not wait for the customer to speak first.)"
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
        await openai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "(The customer has been silent for a while after you finished speaking. Say 'Are you still there?' and then repeat your last question exactly.)"}]
            }
        }))
        await openai_ws.send(json.dumps({"type": "response.create"}))
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
                await handle_user_address_response(stream_sid, ut)

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


async def handle_ai_commands(stream_sid: str, message: str):
    """
    Robust extraction that handles speech-to-speech variations,
    spelling errors, and format inconsistencies
    """
    # # Normalize message for status detection
    normalized_msg = message

    print(":::::::::::::::::::::::", normalized_msg)
    if re.search(r"\bTAG_ADDRESS_REJECT\b", normalized_msg, re.I):
        if stream_sid in call_context:
            call_context[stream_sid].update({
                "addressConfirmed": False,
                "comments": "Address is wrong",
                "slotSelected": False,
                "address_reject_pending_disconnect": False,
            })
        t = address_reject_fallback_tasks.pop(stream_sid, None)
        if t and not t.done():
            t.cancel()
        await cleanup_connections(stream_sid)
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
        await cleanup_connections(stream_sid)
        return

    # Detect booking status (avoid CONFIRMING — matches "Thanks for confirming" on reschedule closings)
    status = None
    if re.search(r"\bCONFIRMED\b", normalized_msg, re.I):
        status = "Confirmed"
    elif re.search(r"\b(DECLINE|DECLINED|DECLINING)\b", normalized_msg, re.I):
        status = "Declined"

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
    2.  **Extract the Date:** Identify any mention of a date. Convert it to an absolute date in `YYYY:MM:DD` format.
    3.  **Extract the Time:** Identify any mention of a time or time range. Convert it to a `HH:mm–HH:mm` format using a 24-hour clock.
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
            
            
            # Update your context with the extracted data
            if stream_sid in call_context:
                upd: Dict[str, Any] = {
                    "slotSelected": True if date_out and time_out else False,
                    "selectedDate": date_out,
                    "selectedSlot": re.sub(r'[–—‒−]', '-', str(time_out)),
                    "comments": comments_out,
                    "sentiment": sentiment_out,
                }
                if merged_data.get("status") == "confirmed" and date_out and time_out:
                    upd["addressConfirmed"] = True
                    upd["comments"] = ""
                call_context[stream_sid].update(upd)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM: {e}")
            # Add fallback logic here if needed
            
        # await asyncio.sleep(10)
        await cleanup_connections(stream_sid)




async def cleanup_connections(stream_sid: Optional[str]):
    if not stream_sid:
        return

    if stream_sid not in cleanup_locks:
        cleanup_locks[stream_sid] = asyncio.Lock()

    async with cleanup_locks[stream_sid]:
        ft = address_reject_fallback_tasks.pop(stream_sid, None)
        if ft and not ft.done():
            ft.cancel()

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
                slotSelected=bool(context.get('slotSelected')),
                selectedDate=context.get('selectedDate'),
                selectedSlot=context.get('selectedSlot'),
                comments=_callback_comments(context),
                sentiment=context.get('sentiment'),
                addressConfirmed=context.get('addressConfirmed'),
                isReschedule=bool(context.get('isReschedule')),
            )
            logger.info(f"Final report: {result.model_dump_json(indent=2)}")

            callback_url = context.get('callbackUrl')
            if callback_url:
                try:
                    async with httpx.AsyncClient() as client:
                        logger.info(f"Sending final report to callback URL: {callback_url}")

                        auth = httpx.BasicAuth(
                            'e1b72f9c-3d54-45c1-8f62-94c7a6b2e718',
                            'c5f1d8a3-1d4b-46f2-9b8c-73f2e2d9a8b7'
                        )

                        response = await client.post(
                            callback_url,
                            json=result.model_dump(),
                            auth=auth,
                            timeout=15.0
                        )

                        if response.status_code == 200:
                            logger.info(f"✅ Payload successfully sent to {callback_url}")
                        else:
                            logger.error(
                                f"❌ Failed to send payload. "
                                f"Status: {response.status_code}, Response: {response.text}"
                            )

                except Exception as e:
                    logger.error(f"Failed to send result to callback URL {callback_url}: {e}")

            del call_context[stream_sid]

        if stream_sid in sender_tasks:
            sender_tasks[stream_sid].cancel()
            del sender_tasks[stream_sid]

        if stream_sid in silence_timer_tasks:
            silence_timer_tasks[stream_sid].cancel()
            silence_timer_tasks.pop(stream_sid, None)
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

        for buf_dict in [audio_buffers, outbound_audio_buffers, ai_transcripts]:
            buf_dict.pop(stream_sid, None)

        cleanup_locks.pop(stream_sid, None)
        logger.info(f"All resources for stream {stream_sid} have been cleaned up.")