from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AttendanceSession
from app.services.attendance import calculate_attendance, export_attendance_csv

router = APIRouter(prefix="/attendance", tags=["attendance"])


@router.get("/{session_id}")
def get_attendance(session_id: int, db: Session = Depends(get_db)):
    """Get the attendance report for a session."""
    session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    report = calculate_attendance(session_id, db)
    return {
        "session_id": session_id,
        "meeting_url": session.meeting_url,
        "started_at": session.started_at.isoformat(),
        "report": report,
    }


@router.get("/{session_id}/export")
def export_attendance(session_id: int, db: Session = Depends(get_db)):
    """Download the attendance report as a CSV file."""
    session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    csv_content = export_attendance_csv(session_id, db)
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_session_{session_id}.csv"},
    )


@router.get("/")
def list_sessions(db: Session = Depends(get_db)):
    """List all attendance sessions."""
    sessions = db.query(AttendanceSession).order_by(AttendanceSession.started_at.desc()).all()
    return [
        {
            "id": s.id,
            "meeting_url": s.meeting_url,
            "started_at": s.started_at.isoformat(),
            "ended_at": s.ended_at.isoformat() if s.ended_at else None,
        }
        for s in sessions
    ]
