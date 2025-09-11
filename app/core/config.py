import pydantic_settings

class Settings(pydantic_settings.BaseSettings):
    MODELS_DIR: str
    AMB_CLASS_PATH: str
    MS_CLASS_PATH: str
    MS_SPLIT_PATH: str
    CLASS_TOKENIZER: str
    MODEL_TOKENIZER: str
    SECRET_KEY: str
    TOKEN: str

    class Config:
        env_file = ".env"

settings = Settings()