import sys

from regimetry.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


def error_message_detail(error, error_detail: sys = None):
    """
    Captures details about the error, including the file name, line number, and error message.
    """
    if error_detail is None:
        return f"{error}"
    exc_info = error_detail.exc_info()
    if exc_info is None or exc_info[2] is None:
        return f"{error}"

    # Get the traceback object from the error detail
    _, _, exc_tb = error_detail.exc_info()
    # Extract the file name and line number where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    # Format the error message
    error_message = "Error occurred in Python script name [{0}] at line number [{1}] with error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    """
    A custom exception class that provides detailed error messages,
    including the script name, line number, and error details.
    """

    def __init__(self, error_message, error_detail: sys = None):
        # Store the original exception
        self.original_exception = error_message
        # Generate the detailed error message
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )
        # Call the base class constructor with the detailed message
        super().__init__(self.error_message)

    def __str__(self):
        """
        Returns the detailed error message when the exception is converted to a string.
        """
        return self.error_message


# Test log messages (optional for debugging)
if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.error("Divide by Zero")
        raise CustomException(e, sys) from e
