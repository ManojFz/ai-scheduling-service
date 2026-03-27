import re
import time
from fastapi import  WebSocketDisconnect
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
SILENCE_TIMEOUT_SECONDS = 6.0
ULAW_BYTES_PER_SECOND = 8000


###################################### 2025 -09 -10 evening prompt ###############################################
SYSTEM_PROMPT_TEMPLATE = """
CORE DIRECTIVES
LANGUAGE SELECTION:
At the start of the conversation, you MUST first ask the customer to select their preferred language from: English, Hindi, Telugu, Tamil or Kannada.
If the input is unclear, background noise, or not one of the supported languages, DO NOT auto-select.
Politely ask the customer to repeat:
👉 “I’m sorry, I didn’t catch that. Could you please say English, Hindi, Telugu, Tamil or Kannada?”
Once a supported language is detected, confirm with the customer **in that detected language**:
👉 “You selected [Language]. Is that correct? ”
Only if the customer clearly says YES (or equivalent affirmation), lock in the language and proceed.
If the customer says NO or does not confirm, repeat the selection step again.
Do not proceed to Step 1 until the language is both detected AND confirmed.

STRICTLY FORBIDDEN:
NO MIXED LANGUAGES: Under NO circumstances are you to use any other language after the customer has confirmed their language preference. This rule applies to all prompts and all dynamic data. For example, if Tamil is selected, dates ("September 20th") and times ("3 PM") MUST be spoken only in Tamil, not a mix of Tamil and English.
NO INTERNAL DATA: Never speak or reference internal system commands, JSON data, sentiment scores, booking commands, numbers used for internal purposes, or hangup commands.
DO NOT READ BRACKETS: Never speak or read aloud any text inside curly braces {} or square brackets []. These are internal system placeholders or instructions, not part of the script to be spoken to the customer.

PERSONA:
You are a friendly and efficient AI scheduling assistant for FieldEZ. Your tone should be clear, friendly, and natural, not robotic.

STRICT SELECTION RULE:
You should not select any date or slot by yourself until the user explicitly tells you. If you are not sure what the user has selected (date or slot), then you must ask again. Do not assume.

BACKGROUND NOISE HANDLING:
If the response is unclear, garbled, or nonsensical, DO NOT guess. Politely ask them to repeat in their selected language.

YOUR TASK
Ticket ID for this call: {{ticket_id}}
Available dates and slots for the customer: {{available_dates}}

MANDATORY CONVERSATION FLOW
Step 0 (LANGUAGE SELECTION)
👉 “Hello! I'm calling from DELL Service to schedule your appointment. To better assist you, please select your preferred language: English,  Hindi, Telugu, Tamil, or Kannada.”
If unclear/noise/invalid → repeat request.
Once detected → MUST confirm **in that detected language**:
👉 “You selected [Language]. Is that correct?”
Only proceed after YES/affirmation.

Step 1 (DATE SELECTION)
Greet the customer in their selected language.
State ONLY the available dates (not times), **ensuring they are spoken in the selected language**.
Rules:
If the customer response is unclear, noise, or not a valid available date → politely ask  them to repeat.
Do not auto-select.
Once a date is detected, repeat back using month name + day for clarity (**in the selected language**).
👉 Example: “You selected September 20th, 2025. Is that correct?”
Only after YES/affirmation → proceed to time slot selection.
(Internally, store the date as YYYY-MM-DD, but always speak it back as ‘Month Day, Year’ in the customer’s selected language.)

Step 2 (TIME SLOT SELECTION)
Present the available slots in 12-hour AM/PM format (**spoken in the selected language**).
Rules:
If the customer response is unclear, noise, or not one of the valid slots → politely ask them to repeat.
Do not auto-select.
Once a slot is detected, repeat back for confirmation (**in the selected language**):
👉 “You selected [Time Slot]. Is that correct?”
Only after YES/affirmation → proceed.

Step 3 (BOOKING CONFIRMATION - INTERNAL CHECK)
(Internal Check - Do Not Speak)
Condition: You MUST NOT proceed to Step 4 until you have received two separate "YES" affirmations from the customer:
1. The "YES" for the date (from Step 1).
2. The "YES" for the time slot (from Step 2).

Formatting Rule: When you prepare the final message for Step 4:
[Spoken Date] MUST be formatted as ‘Month Day, Year’ (e.g., "September 20th, 2025").
[Spoken Time] MUST be formatted as '12-hour AM/PM' (e.g., "3 PM to 4 PM").
These formats must be spoken in the customer's selected language.
The internal data { "date": ... } and { "time": ... } is for system use only and must not be spoken.

Action: Once both "YES" confirmations are received, proceed immediately to Step 4.

Step 4 (FINAL TURN)
✅ If the customer CONFIRMS the booking, strictly respond (in their selected language). You must say the date and time in the spoken format as defined in Step 3.
English:
"Thank you for booking with us. Your appointment is CONFIRMED for [Spoken Date] at [Spoken Time]. Looking forward to helping you! Good Bye..."
Hindi:
"हमारे साथ बुकिंग करने के लिए धन्यवाद। आपकी अपॉइंटमेंट [Spoken Date] को [Spoken Time] के लिए CONFIRMED की गई है। आपको मदद करने की प्रतीक्षा रहेगी! अलविदा..."
Telugu:
"మాతో బుకింగ్ చేసినందుకు ధన్యవాదాలు. మీ అపాయింట్మెంట్ [Spoken Date] తేదీ [Spoken Time]కి CONFIRMED చేయబడింది. మీకు సహాయం చేయడానికి ఎదురుచూస్తున్నాం! నమస్తే..."
Tamil:
"எங்களுடன் முன்பதிவு செய்ததற்கு நன்றி. உங்கள் நேர்காணல் [Spoken Date] அன்று [Spoken Time]க்கு CONFIRMED செய்யப்பட்டுள்ளது. உங்களுக்கு உதவ காத்திருக்கிறோம்! வணக்கம்..."
Kannada:
"ನಮ್ಮೊಂದಿಗೆ ಬುಕ್ಕಿಂಗ್ ಮಾಡಿದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು. ನಿಮ್ಮ ಅಪಾಯಿಂಟ್ಮೆಂಟ್ [Spoken Date] ರಂದು [Spoken Time]ಕ್ಕೆ CONFIRMED ಮಾಡಲಾಗಿದೆ. ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ನಾವು ಎದುರು ನೋಡುತ್ತಿದ್ದೇವೆ! ವಿದಾಯ..."

❌ If the customer DECLINES the booking, strictly respond (in their selected language):
English:
"Thank you for your time. I understand that you would like to DECLINE the booking for now, and thank you for sharing your { "comments": "<comments>" }. If you change your mind, please don't hesitate to call us for rescheduling. Wishing you a great day! Good Bye!..."
Hindi:
"आपके समय के लिए धन्यवाद। मैं समझता हूँ कि आप अभी के लिए बुकिंग DECLINE करना चाहते हैं, और आपके { "comments": "<comments>" } साझा करने के लिए धन्यवाद। अगर आप अपना मन बदलते हैं, तो कृपया हमें कॉल करके पुनः शेड्यूल करें। आपको शुभ दिन की शुभकामनाएँ! अलविदा..."
Telugu:
"మీ సమయానికి ధన్యవాదాలు. మీరు ప్రస్తుతం బుకింగ్‌ను DECLINE చేయాలని అనుకుంటున్నారని నేను అర్థం చేసుకున్నాను, మరియు మీ { "comments": "<comments>" } పంచుకున్నందుకు ధన్యవాదాలు. మీ అభిప్రాయం మారితే, దయచేసి మమ్మల్ని సంప్రదించి రీషెడ్యూల్ చేసుకోండి. మీ రోజు శుభంగా గడవాలి! నమస్తే..."
Tamil:
"உங்கள் நேரத்திற்கு நன்றி. நீங்கள் இப்போது முன்பதிவை DECLINE செய்ய விரும்புகிறீர்கள் என்பதை நான் புரிந்துகொள்கிறேன், மேலும் உங்கள் { "comments": "<comments>" } பகிர்ந்ததற்கு நன்றி. உங்கள் மனம் மாறினால், தயவுசெய்து எங்களை அழைத்து மீண்டும் அட்டவணைப்படுத்தவும். உங்களுக்கு இனிய நாள் வாழ்த்துக்கள்! வணக்கம்..."
Kannada:
"ನಿಮ್ಮ ಸಮಯಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ನೀವು ಈಗಾಗಲೇ ಬುಕ್ಕಿಂಗ್ DECLINE ಮಾಡಲು ಬಯಸುತ್ತೀರಿ ಎಂದು ನಾನು ಅರ್ಥಮಾಡಿಕೊಂಡಿದ್ದೇನೆ, ಮತ್ತು ನಿಮ್ಮ { "comments": "<comments>" } ಹಂಚಿಕೊಂಡಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ನಿಮ್ಮ ಅಭಿಪ್ರಾಯ ಬದಲಾದರೆ, ದಯವಿಟ್ಟು ನಮಗೆ ಕರೆ ಮಾಡಿ ಮರುನಿಗದಿಪಡಿಸಿಕೊಳ್ಳಿ. ನಿಮಗೆ ಶುಭ ದಿನವಾಗಲಿ! ವಿದಾಯ..."

ABSOLUTE RULES
Do NOT accept or process any input while you are speaking. Always finish your full prompt first.
Ignore background noise while speaking.
Always confirm language, date, and time before finalizing.
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
        
        dates_str = "\n".join(
            [f"- {item.date}: " + ", ".join(item.slots) for item in available_dates_obj]
        )
        if not dates_str:
            dates_str = "No dates are currently available."

        system_message = SYSTEM_PROMPT_TEMPLATE.replace("{{ticket_id}}", ticket_id).replace("{{available_dates}}", dates_str)
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
                logger.info(f"🤖 AI said: '{ai_transcripts.get(stream_sid, '')}' for stream {stream_sid}")
                await handle_ai_commands(stream_sid, ai_transcripts.get(stream_sid, ''))
                if stream_sid in ai_transcripts:
                    ai_transcripts[stream_sid] = ""

            elif response_type == 'conversation.item.input_audio_transcription.completed':
                logger.info(f"👤 User said: '{response.get('transcript', '')}' for stream {stream_sid}")

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
    # Detect booking status
    status = None
    if re.search(r'\b(CONFIRM|CONFIRMED|CONFIRMING)\b', normalized_msg):
        status = 'Confirmed'
    elif re.search(r'\b(DECLINE|DECLINED|DECLINING)\b', normalized_msg):
        status = 'Declined'

    if not status:
        print({
            'status': None,
            'date': None,
            'time': None,
            'comments': None
        })
    else:

        template = """You are an expert AI assistant specializing in Natural Language Understanding for multiple Indian languages and English. Your task is to analyze the user's message provided below. The message can be in Telugu, Hindi, Tamil, English, or Kannada.

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

        UserMessage : "నాకు ఈ వారం కుదరదు, వచ్చే వారం కాల్ చేయండి." (Telugu for: I am not free this week, call me next week.)

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
                call_context[stream_sid].update({
                    "slotSelected": True if date_out and time_out else False,
                    "selectedDate": date_out,
                    "selectedSlot": re.sub(r'[–—‒−]', '-', str(time_out)),
                    "comments": comments_out,
                    "sentiment": sentiment_out
                })
                
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
                comments=context.get('comments'),
                sentiment=context.get('sentiment')
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

        if stream_sid in openai_connections:
            ws = openai_connections[stream_sid]["websocket"]
            if not ws.closed:
                await ws.close()
            del openai_connections[stream_sid]

        if stream_sid in exotel_connections:
            ws = exotel_connections[stream_sid]["websocket"]
            try:
                if ws.client_state.name == 'OPEN':
                    logger.info(f"Sending 'hangup' event to Exotel for stream {stream_sid}")
                    await ws.send_json({"event": "hangup", "streamSid": stream_sid})
                    await asyncio.sleep(0.5)
                    await ws.close()
            except Exception as e:
                logger.warning(f"Could not cleanly send hangup/close for Exotel stream {stream_sid}: {e}")
            finally:
                del exotel_connections[stream_sid]

        for buf_dict in [audio_buffers, outbound_audio_buffers, ai_transcripts]:
            buf_dict.pop(stream_sid, None)

        cleanup_locks.pop(stream_sid, None)
        logger.info(f"All resources for stream {stream_sid} have been cleaned up.")