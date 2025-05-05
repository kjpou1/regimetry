from argparse import ArgumentParser

from regimetry.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class LoggingArgumentParser(ArgumentParser):
    """
    Custom ArgumentParser that logs errors instead of printing to stderr.
    """

    def error(self, message: str):
        logging.error("Argument parsing error: %s", message)
        self.print_help()
        raise SystemExit(2)
