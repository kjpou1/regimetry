import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

from regimetry.config.config import Config


class LoggerManager:
    """Custom Logger Manager with enhanced features."""

    # Initialize default log settings
    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    LOGS_DIR = Config().LOG_DIR
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_JSON = os.getenv("LOG_JSON", "false").lower() == "true"

    # Ensure the logs directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)
    LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

    # Shared formatter for plain text logs
    PLAIN_FORMATTER = logging.Formatter(
        "[ %(asctime)s ] %(levelname)s [%(name)s:%(lineno)d] - %(message)s"
    )

    # Shared formatter for JSON logs
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "logger": record.name,
                "line": record.lineno,
                "message": record.getMessage(),
            }
            return json.dumps(log_record)

    @classmethod
    def set_log_level(cls, level):
        """Dynamically set the logging level."""
        cls.LOG_LEVEL = level
        logging.getLogger().setLevel(level)
        for handler in logging.getLogger().handlers:
            handler.setLevel(level)

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger with the specified name and enhanced configuration."""
        logger = logging.getLogger(name)
        logger.setLevel(LoggerManager.LOG_LEVEL)

        if not logger.hasHandlers():
            # Add a rotating file handler
            file_handler = RotatingFileHandler(
                LoggerManager.LOG_FILE_PATH, maxBytes=5 * 1024 * 1024, backupCount=5
            )
            file_handler.setFormatter(
                LoggerManager.JsonFormatter()
                if LoggerManager.LOG_JSON
                else LoggerManager.PLAIN_FORMATTER
            )
            logger.addHandler(file_handler)

            # Add a console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                LoggerManager.JsonFormatter()
                if LoggerManager.LOG_JSON
                else LoggerManager.PLAIN_FORMATTER
            )
            logger.addHandler(console_handler)

        return logger
