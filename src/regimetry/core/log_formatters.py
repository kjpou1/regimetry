import logging
import os


class RelativePathFormatter(logging.Formatter):
    def format(self, record):
        # Convert absolute path to relative path from project root
        if record.pathname:
            base_path = os.path.abspath(os.getcwd())
            relative_path = os.path.relpath(record.pathname, base_path)
            record.relativepath = relative_path.replace("\\", "/")
        else:
            record.relativepath = record.name

        return super().format(record)

