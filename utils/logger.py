# tools/logger.py
import os
import sys
from loguru import logger

def init_logger():


    os.makedirs("logs", exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<italic>{line}</italic> - <level>{message}</level>",
        level="DEBUG",
        colorize=True
    )

    logger.add(
        "logs/athena_execution.log", # путь к файлу логов
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="10 MB",
        retention="5 days",
        compression="zip",
        encoding="utf-8"
    )


init_logger()
__all__ = ['logger']