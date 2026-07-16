import json
import httpx
import asyncio
from fastapi import FastAPI, Request, HTTPException,  WebSocket, WebSocketDisconnect
from fastapi.responses import  JSONResponse
from service import *



# --- FastAPI App Initialization ---
app = FastAPI()



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app instance
app = FastAPI(title="Schedule API")





# PUBLIC_WS_BASE = "https://fieldez-hjbzfyhjb6dsdsdw.centralindia-01.azurewebsites.net"
# PUBLIC_WS_BASE = "https://fieldeztata-poc-a4abeebch0h2hrcz.centralindia-01.azurewebsites.net"
PUBLIC_WS_BASE = "https://jameson-nondiscriminative-zaiden.ngrok-free.dev"



@app.post("/initiate-schedule-call")
async def initiate_schedule_call(payload: ScheduleCallRequest):
    print("payload::::::::", payload)
    url = f"https://{EXOTEL_SUBDOMAIN}/v1/Accounts/{EXOTEL_SID}/Calls/connect.json"
    flow_url = f"http://my.exotel.com/{EXOTEL_SID}/exoml/start_voice/{EXOTEL_FLOW_APP_ID}"
   
    custom_field_data = {
        "name": f"Customer-{payload.customerPhone[-4:]}",
        "ticketId": payload.ticketId,
    }
   
    form_data = {
        "From": payload.customerPhone,
        "CallerId": EXOTEL_CALLER_ID,
        "Url": flow_url,
        "CallType": "trans",
        "CustomField": json.dumps(custom_field_data),
        "TimeLimit":5,
        "StatusCallback": f"{PUBLIC_WS_BASE}/exotel-webhook"
    }
 
    auth = (EXOTEL_API_KEY, EXOTEL_API_TOKEN)
   
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, data=form_data, auth=auth)
            resp.raise_for_status()
           
            call_details = resp.json()
            call_sid = call_details.get("Call", {}).get("Sid")
 
            if not call_sid:
                raise HTTPException(status_code=500, detail="Failed to get CallSid from Exotel.")
 
            logger.info(f"Call initiated with CallSid: {call_sid}")
           
            ctx = {
                "ticketId": payload.ticketId,
                "callbackUrl": str(payload.callbackUrl),
                "serviceTag": str(payload.serviceTag).strip(),
                "serviceTagConfirmed": None,
                "address": summarize_address_for_speech(payload.address),
                "pincode": getattr(payload, "pincode", None),
                "availableDates": payload.availableDates,
                "callConnected": True,
                "isLineBusy": False,
                "slotSelected": False,
                "selectedDate": None,
                "selectedSlot": None,
                "comments": "",
                "sentiment": None,
                "addressConfirmed": None,
                "last_assistant_message": "",
                "status": "active",
                "isReschedule": False,
                "slotTier": None,
                "proximityRejected": False,
                "confirmedOfferDate": None,
            }
            call_context[call_sid] = ctx
            call_context[f"ticket:{str(payload.ticketId).strip()}"] = ctx
 
            return JSONResponse(status_code=resp.status_code, content={"result": call_details, "status": "success"})
    except Exception as e:
        logger.error(f"Error starting call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error starting call: {str(e)}")


@app.post("/prepare-inbound-call")
async def prepare_inbound_call(payload: PrepareInboundCallRequest):
    call_sid = str(payload.callSid).strip()
    ticket_id = str(payload.ticketId).strip()
    if not call_sid:
        raise HTTPException(status_code=400, detail="callSid is required.")
    if not ticket_id:
        raise HTTPException(status_code=400, detail="ticketId is required.")

    ctx = {
        "ticketId": payload.ticketId,
        "callbackUrl": str(payload.callbackUrl),
        "serviceTag": str(getattr(payload, "serviceTag", "") or "").strip(),
        "serviceTagConfirmed": None,
        "address": summarize_address_for_speech(payload.address),
        "pincode": getattr(payload, "pincode", None),
        "availableDates": payload.availableDates,
        "callConnected": True,
        "isLineBusy": False,
        "slotSelected": False,
        "selectedDate": None,
        "selectedSlot": None,
        "comments": "",
        "sentiment": None,
        "addressConfirmed": None,
        "last_assistant_message": "",
        "status": "active",
        "isReschedule": False,
        "inbound": True,
        "customerPhone": payload.customerPhone,
        "slotTier": None,
        "proximityRejected": False,
        "confirmedOfferDate": None,
    }
    call_context[call_sid] = ctx
    call_context[f"ticket:{ticket_id}"] = ctx

    logger.info(
        "Prepared inbound call context callSid=%s ticketId=%s customerPhone=%s dates=%s",
        call_sid,
        ticket_id,
        payload.customerPhone,
        len(payload.availableDates),
    )
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "callSid": call_sid, "ticketId": ticket_id},
    )



# --- WEBSOCKET AND MEDIA HANDLING LOGIC ---

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    person_name = websocket.query_params.get("name", "Unknown")
    logger.info(f"WebSocket connection accepted for: {person_name} from {websocket.client.host}")
    stream_sid = None
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get('event')
 
            if event == 'start':
                stream_sid = data['start']['stream_sid']
                call_sid = data['start']['call_sid']
                link_stream_sid_to_call_context(
                    stream_sid,
                    call_sid,
                    data,
                    query_ticket_id=websocket.query_params.get("ticketId"),
                )
 
                exotel_connections[stream_sid] = {"websocket": websocket, "call_sid": call_sid}
                audio_buffers[stream_sid] = b""
                ai_transcripts[stream_sid] = ""
                outbound_audio_buffers[stream_sid] = bytearray()
               
                logger.info(f"Exotel stream started: {stream_sid}")
               
                sender_tasks[stream_sid] = asyncio.create_task(paced_audio_sender(stream_sid))
                openai_ws = await start_voice_session(stream_sid, person_name)
                if openai_ws is not None:
                    asyncio.create_task(handle_openai_responses(stream_sid, openai_ws))
 
            elif event == 'media' and stream_sid:
                await handle_exotel_media(stream_sid, data)
 
            elif event == 'stop':
                logger.info(f"Exotel stream ended {stream_sid}: {data.get('stop')}")
                break
    except WebSocketDisconnect:
        logger.info(f"Exotel WebSocket disconnected for {stream_sid}")
    except Exception as e:
        logger.error(f"Error in media stream for {stream_sid}: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up connections for stream {stream_sid}")
        await cleanup_connections(stream_sid)



@app.post("/exotel-webhook")
async def exotel_webhook(request: Request):
    # Log headers first
    headers = dict(request.headers)
    logger.info(f"Webhook received - Headers: {headers}")
   
    # Get content type
    content_type = headers.get("content-type", "")
   
    try:
        # Parse data based on content type (only once!)
        if "application/json" in content_type:
            data = await request.json()
        elif "application/x-www-form-urlencoded" in content_type:
            form_data = await request.form()
            data = dict(form_data)
        else:
            # Try JSON first, then form data as fallback
            try:
                data = await request.json()
            except:
                form_data = await request.form()
                data = dict(form_data)
   
    except Exception as e:
        logger.error(f"Failed to parse webhook data: {e}")
        return JSONResponse(content={"message": "Invalid data format"}, status_code=400)
   
    logger.info(f"Webhook data: {data}")
   
    # Get call status and call_sid using proper field names
    call_status = (data.get("CallStatus") or
                   data.get("DialCallStatus") or
                   data.get("Status", "")).lower()
   
    call_sid = data.get("CallSid") or data.get("Sid")
   
    logger.info(f"Extracted - Status: {call_status}, CallSid: {call_sid}")
   
    # Handle no-answer, busy, or failed calls
    if call_status in ["no-answer", "failed", "busy", "no_answer"]:
        is_line_busy = call_status == "busy"
        # Find the ticket_id and callbackUrl from call_context using call_sid
        ticket_id = None
        callback_url = None
        context_to_update = None

        for context_key, context_value in call_context.items():
            if context_key == call_sid or context_value.get("call_sid") == call_sid:
                ticket_id = context_value.get("ticketId")
                callback_url = context_value.get("callbackUrl")
                context_to_update = context_value
                break

        if context_to_update is not None:
            context_to_update["callConnected"] = False
            context_to_update["isLineBusy"] = is_line_busy

        comments = (
            "Customer line was busy."
            if is_line_busy
            else f"Customer did not answer the call. Status: {call_status}"
        )
        response = {
            "ticketId": ticket_id,
            "callConnected": False,
            "isLineBusy": is_line_busy,
            "slotSelected": False,
            "selectedDate": None,
            "selectedSlot": None,
            "comments": comments,
            "sentiment": 0,
            "addressConfirmed": None,
        }

        if callback_url:
            try:
                async with httpx.AsyncClient() as client:
                    logger.info(
                        "No-answer callback POST starting | ticketId=%s url=%s",
                        ticket_id,
                        callback_url,
                    )
                    auth = httpx.BasicAuth('e1b72f9c-3d54-45c1-8f62-94c7a6b2e718', 'c5f1d8a3-1d4b-46f2-9b8c-73f2e2d9a8b7')
                    http_resp = await client.post(callback_url, json=response, auth=auth, timeout=15.0)
                    body_preview = (http_resp.text or "")[:500]
                    if http_resp.status_code == 200:
                        logger.info(
                            "No-answer callback POST OK | ticketId=%s status=%s url=%s body_preview=%r",
                            ticket_id,
                            http_resp.status_code,
                            callback_url,
                            body_preview,
                        )
                    else:
                        logger.error(
                            "No-answer callback POST non-200 | ticketId=%s status=%s url=%s body=%r",
                            ticket_id,
                            http_resp.status_code,
                            callback_url,
                            body_preview,
                        )
            except Exception as e:
                logger.error(
                    "No-answer callback POST exception | ticketId=%s url=%s error=%s",
                    ticket_id,
                    callback_url,
                    e,
                    exc_info=True,
                )

        logger.info(f"No-answer webhook payload (to Exotel): {response}")
        return JSONResponse(content=response)
 
    # For other statuses, just acknowledge
    logger.info(f"Webhook processed for status: {call_status}")
    return JSONResponse(content={"message": "Webhook processed"}, status_code=200)
 

