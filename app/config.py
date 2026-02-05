from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///./attendance.db"

    # Recall.ai
    recall_api_key: str = ""

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "attendance-embeddings"

    # Face matching: minimum cosine similarity to count as a match
    match_threshold: float = 0.6

    # Attendance: fraction of pings a student must be matched in to be "Present"
    presence_threshold: float = 0.6

    # How often (in minutes) to capture a frame during a meeting
    ping_interval_minutes: int = 5

    model_config = {"env_file": ".env"}


settings = Settings()
