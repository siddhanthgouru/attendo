from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AttendanceSession
from app.services.recall import recall_client, StubRecallClient
from app.services.attendance import process_frame

router = APIRouter(prefix="/meeting", tags=["meeting"])


class MeetingStartRequest(BaseModel):
    meeting_url: str  # Real URL for production, folder path for stub mode


@router.post("/start")
async def start_meeting(request: MeetingStartRequest, db: Session = Depends(get_db)):
    """
    Start an attendance session by dispatching a bot to the meeting.

    In stub mode: pass a folder path containing test images.
    In production: pass a Zoom/Teams/Meet URL.
    """
    # Create an attendance session in the DB
    session = AttendanceSession(meeting_url=request.meeting_url)
    db.add(session)
    db.commit()
    db.refresh(session)

    # Dispatch the bot
    bot_info = await recall_client.dispatch_bot(request.meeting_url)

    # In stub mode, immediately process all frames
    if isinstance(recall_client, StubRecallClient):
        frames = recall_client.get_frames(bot_info["id"])
        for frame in frames:
            process_frame(frame, session.id, db)

    return {
        "session_id": session.id,
        "bot": bot_info,
        "message": "Attendance session started.",
    }


@router.post("/webhook")
async def recall_webhook(payload: dict, db: Session = Depends(get_db)):
    """
    Webhook endpoint for Recall.ai to send events.

    In production, Recall.ai calls this when:
    - A new video frame is available
    - The bot joins/leaves the meeting
    - The meeting ends
    """
    event_type = payload.get("event")

    if event_type == "bot.status_change":
        status = payload.get("data", {}).get("status")
        if status == "done":
            # Meeting ended — mark the session as complete
            bot_id = payload.get("data", {}).get("bot_id")
            # In a real implementation, look up session by bot_id
            return {"message": "Meeting ended, session closed."}

    return {"message": f"Received event: {event_type}"}
