from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import create_tables
from app.routers import attendance, meeting, registration


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before the app begins serving requests."""
    create_tables()
    yield


app = FastAPI(
    title="Attendance Bot API",
    description="Automated meeting attendance using face recognition.",
    version="0.1.0",
    lifespan=lifespan,
)

# Register all routers
app.include_router(registration.router)
app.include_router(meeting.router)
app.include_router(attendance.router)


@app.get("/")
def root():
    return {"message": "Attendance Bot API is running.", "docs": "/docs"}
