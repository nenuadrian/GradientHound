from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    checkpoint_dir: str = "./checkpoints"
    host: str = "0.0.0.0"
    port: int = 8642

    model_config = {"env_prefix": "GHOUND_"}


settings = Settings()
