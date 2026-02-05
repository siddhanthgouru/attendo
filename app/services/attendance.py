"""
Attendance logic with temporal persistence.

Core idea: a student is only marked "Present" if they are recognized
in a significant percentage of captured frames (pings) during the session.
"""
import csv
import io
from datetime import datetime

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.models import AttendanceSession, Ping, Student
from app.services.face import detect_and_embed
from app.services.vector_store import vector_store


def process_frame(image: np.ndarray, session_id: int, db: Session) -> list[dict]:
    """
    Process a single video frame: detect faces, match against registered students,
    and record pings in the database.

    Args:
        image: BGR image as numpy array.
        session_id: The active attendance session ID.
        db: Database session.

    Returns:
        List of match results for this frame.
    """
    faces = detect_and_embed(image)
    results = []

    for face in faces:
        # Query vector store for the best match
        matches = vector_store.query_embedding(face["embedding"], top_k=1)

        if matches and matches[0]["score"] >= settings.match_threshold:
            match = matches[0]
            # Look up the student's DB record
            student = db.query(Student).filter(Student.student_id == match["student_id"]).first()
            if student:
                ping = Ping(
                    session_id=session_id,
                    student_id=student.id,
                    matched=True,
                    confidence=match["score"],
                )
                db.add(ping)
                results.append({
                    "student_id": student.student_id,
                    "name": student.name,
                    "confidence": match["score"],
                    "matched": True,
                })

    db.commit()
    return results


def calculate_attendance(session_id: int, db: Session) -> list[dict]:
    """
    Calculate final attendance for a session using temporal persistence.

    A student is "Present" if they were matched in >= PRESENCE_THRESHOLD
    fraction of the total pings for that session.

    Returns:
        List of dicts with student info and attendance status.
    """
    # Get all pings for this session
    pings = db.query(Ping).filter(Ping.session_id == session_id).all()
    if not pings:
        return []

    # Count total unique timestamps (i.e., how many frames were captured)
    unique_timestamps = set()
    for p in pings:
        # Round to the nearest minute to group pings from the same frame capture
        rounded = p.timestamp.replace(second=0, microsecond=0)
        unique_timestamps.add(rounded)
    total_pings = max(len(unique_timestamps), 1)

    # Count how many pings each student was matched in
    student_counts: dict[int, int] = {}
    student_confidences: dict[int, list[float]] = {}
    for p in pings:
        if p.matched:
            student_counts[p.student_id] = student_counts.get(p.student_id, 0) + 1
            student_confidences.setdefault(p.student_id, []).append(p.confidence)

    # Build the attendance report
    all_students = db.query(Student).all()
    report = []
    for student in all_students:
        match_count = student_counts.get(student.id, 0)
        presence_ratio = match_count / total_pings
        avg_confidence = (
            sum(student_confidences.get(student.id, [])) / match_count
            if match_count > 0
            else 0.0
        )

        report.append({
            "student_id": student.student_id,
            "name": student.name,
            "status": "Present" if presence_ratio >= settings.presence_threshold else "Absent",
            "pings_matched": match_count,
            "total_pings": total_pings,
            "presence_ratio": round(presence_ratio, 2),
            "avg_confidence": round(avg_confidence, 3),
        })

    return report


def export_attendance_csv(session_id: int, db: Session) -> str:
    """Generate a CSV string from the attendance report."""
    report = calculate_attendance(session_id, db)
    session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student ID", "Name", "Status", "Pings Matched", "Total Pings", "Presence Ratio", "Avg Confidence"])

    for row in report:
        writer.writerow([
            row["student_id"],
            row["name"],
            row["status"],
            row["pings_matched"],
            row["total_pings"],
            row["presence_ratio"],
            row["avg_confidence"],
        ])

    return output.getvalue()
