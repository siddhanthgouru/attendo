"""
Recall.ai integration service.

When RECALL_API_KEY is set, dispatches real bots to meetings.
Otherwise, provides a stub that simulates meetings using local images.
"""
import glob
import os

import httpx

from app.config import settings
from app.services.face import load_image


RECALL_API_BASE = "https://us-west-2.recall.ai/api/v1"


class RecallClient:
    """Real Recall.ai API client."""

    def __init__(self):
        self._headers = {
            "Authorization": f"Token {settings.recall_api_key}",
            "Content-Type": "application/json",
        }

    async def dispatch_bot(self, meeting_url: str) -> dict:
        """Send a bot to join a meeting. Returns the bot info from Recall.ai."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{RECALL_API_BASE}/bot",
                headers=self._headers,
                json={
                    "meeting_url": meeting_url,
                    "bot_name": "Attendance Bot",
                    "real_time_media": {
                        "websocket_video_output_enabled": True,
                    },
                },
            )
            response.raise_for_status()
            return response.json()

    async def get_bot_status(self, bot_id: str) -> dict:
        """Check the status of a deployed bot."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{RECALL_API_BASE}/bot/{bot_id}",
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()


class StubRecallClient:
    """
    Simulated Recall.ai client for local development.

    Instead of joining a real meeting, it reads images from a folder
    to simulate video frames. Place test images in a folder and pass
    the folder path as the "meeting_url".
    """

    def __init__(self):
        self._sessions: dict[str, list] = {}

    async def dispatch_bot(self, meeting_url: str) -> dict:
        """
        In stub mode, 'meeting_url' is a path to a folder of images.
        Each image simulates one participant's video frame at one ping.
        """
        bot_id = f"stub-bot-{len(self._sessions) + 1}"

        # Load all images from the folder
        frames = []
        if os.path.isdir(meeting_url):
            patterns = ["*.jpg", "*.jpeg", "*.png"]
            for pattern in patterns:
                for path in sorted(glob.glob(os.path.join(meeting_url, pattern))):
                    frames.append(load_image(path))

        self._sessions[bot_id] = frames

        return {
            "id": bot_id,
            "status": "active",
            "frame_count": len(frames),
            "message": f"Stub bot created. Found {len(frames)} test frames.",
        }

    def get_frames(self, bot_id: str) -> list:
        """Get the simulated frames for a stub bot."""
        return self._sessions.get(bot_id, [])

    async def get_bot_status(self, bot_id: str) -> dict:
        return {"id": bot_id, "status": "done"}


def get_recall_client() -> RecallClient | StubRecallClient:
    if settings.recall_api_key and settings.recall_api_key != "your_recall_api_key_here":
        return RecallClient()
    return StubRecallClient()


recall_client = get_recall_client()
