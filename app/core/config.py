import pydantic_settings

class Settings(pydantic_settings.BaseSettings):
    model_path: str
    tokenizer: str
    class_path: str
    mongodb_url: str
    mongodb_db_name: str

    class Config:
        env_file = ".env"

settings = Settings()