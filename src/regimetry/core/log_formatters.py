import os
import logging

class RelativePathFormatter(logging.Formatter):
    """
    A custom log formatter that replaces %(name)s with a relative file path,
    enabling clickable log links in modern IDEs (e.g., VSCode).

    This formatter extracts the absolute `pathname` of the log record,
    calculates its relative path from the current working directory, and
    injects it as `record.relativepath`. This allows logging output like:

        [regimetry/host.py:56] - Executing data ingestion workflow.

    instead of:

        [regimetry.host:56] - ...

    Why this matters:
    -----------------
    IDEs like VSCode and PyCharm support clickable file/line references
    in log output. However, they require paths in the format:
        [relative/path/to/file.py:lineno]

    If you use just the logger name (e.g., `regimetry.host`), the link
    won't work because it’s a module name, not a file path.

    This formatter ensures all logs use `[file.py:lineno]` format
    while maintaining full control over the display.

    Example usage in a logger:
        formatter = RelativePathFormatter(
            "[ %(asctime)s ] %(levelname)s [%(relativepath)s:%(lineno)d] - %(message)s"
        )
    """

    def format(self, record):
        if record.pathname:
            base_path = os.path.abspath(os.getcwd())
            relative_path = os.path.relpath(record.pathname, base_path)
            record.relativepath = relative_path.replace("\\", "/")
        else:
            record.relativepath = record.name

        return super().format(record)
