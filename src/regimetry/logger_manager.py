import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

from regimetry.config.config import Config
from regimetry.core.log_formatters import RelativePathFormatter

class LoggerManager:
    """Custom Logger Manager with enhanced features."""

    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_JSON = os.getenv("LOG_JSON", "false").lower() == "true"

    # Shared formatter for plain text logs
    PLAIN_FORMATTER = RelativePathFormatter(
        "[ %(asctime)s ] %(levelname)s [%(relativepath)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
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
        logger.propagate = False

        if not logger.hasHandlers():
            config = Config()
            log_dir = config.LOG_DIR
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, LoggerManager.LOG_FILE)

            # File handler
            file_handler = RotatingFileHandler(
                log_path, maxBytes=5 * 1024 * 1024, backupCount=5
            )
            formatter = LoggerManager.JsonFormatter() if LoggerManager.LOG_JSON else LoggerManager.PLAIN_FORMATTER
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger
