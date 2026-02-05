import os
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Student
from app.services.face import get_single_embedding, load_image_from_bytes
from app.services.vector_store import vector_store

router = APIRouter(prefix="/register", tags=["registration"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")


@router.post("/")
async def register_student(
    name: str = Form(...),
    student_id: str = Form(...),
    photo: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Register a new student for attendance tracking.

    1. Uploads their selfie
    2. Detects exactly one face
    3. Generates a 512-d embedding
    4. Stores embedding in vector DB
    5. Saves student record in SQLite
    """
    # Check if student_id already exists
    existing = db.query(Student).filter(Student.student_id == student_id).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Student ID '{student_id}' is already registered.")

    # Read and validate the uploaded photo
    photo_bytes = await photo.read()
    if not photo_bytes:
        raise HTTPException(status_code=400, detail="Uploaded photo is empty.")

    image = load_image_from_bytes(photo_bytes)

    # Detect face and generate embedding
    try:
        embedding = get_single_embedding(image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save the photo to disk
    file_ext = os.path.splitext(photo.filename or "photo.jpg")[1] or ".jpg"
    filename = f"{student_id}_{uuid.uuid4().hex[:8]}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(photo_bytes)

    # Store embedding in vector database
    vector_store.store_embedding(student_id, embedding)

    # Create database record
    student = Student(
        name=name,
        student_id=student_id,
        embedding_id=student_id,  # same ID used in vector store
        photo_path=file_path,
    )
    db.add(student)
    db.commit()
    db.refresh(student)

    return {
        "message": f"Student '{name}' registered successfully.",
        "student_id": student.student_id,
        "db_id": student.id,
    }


@router.get("/students")
def list_students(db: Session = Depends(get_db)):
    """List all registered students."""
    students = db.query(Student).all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "student_id": s.student_id,
            "created_at": s.created_at.isoformat(),
        }
        for s in students
    ]


@router.delete("/{student_id}")
def delete_student(student_id: str, db: Session = Depends(get_db)):
    """Delete a student and their embedding."""
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    # Remove embedding from vector store
    vector_store.delete_embedding(student_id)

    # Remove photo file
    if os.path.exists(student.photo_path):
        os.remove(student.photo_path)

    db.delete(student)
    db.commit()
    return {"message": f"Student '{student.name}' deleted."}
