"""
Launch the asynchronous system.

This script initializes the application by parsing command-line arguments,
creating a Host instance, and launching its main logic asynchronously.
"""

import asyncio

from regimetry.host import Host
from regimetry.logger_manager import LoggerManager
from regimetry.runtime.command_line import CommandLine

logging = LoggerManager.get_logger(__name__)


async def launch_async():
    """
    Main asynchronous launch point.

    Parses command-line arguments, initializes the Host instance, and
    launches the main logic asynchronously.
    """
    try:
        args = CommandLine.parse_arguments()
        logging.info("Launching host with arguments: %s", args)

        if args.debug:
            LoggerManager.set_log_level(logging.debug)

        # Create an instance of Host with parsed arguments
        instance = Host(args)

        # Launch the async main function with the parsed arguments
        await instance.run_async()
    except ValueError as e:
        logging.error("ValueError: %s", e)
    except KeyboardInterrupt:
        logging.info("Execution interrupted by user.")
    except Exception as e:
        logging.error("Unexpected error occurred: %s", e)


def launch():
    asyncio.run(launch_async())


if __name__ == "__main__":
    asyncio.run(launch_async())
