import argparse

from regimetry.models.command_line_args import CommandLineArgs

from .logging_argument_parser import LoggingArgumentParser


class CommandLine:
    @staticmethod
    def parse_arguments() -> CommandLineArgs:
        """
        Parse command-line arguments and return a CommandLineArgs object.

        Supports subcommands like 'ingest' and 'train'.
        """
        parser = LoggingArgumentParser(description="Frostfire Chart Sifter Application")

        # Create subparsers for subcommands
        subparsers = parser.add_subparsers(dest="command", help="Subcommands")

        # Subcommand: ingest
        ingest_parser = subparsers.add_parser(
            "ingest", help="Run ingestion pipeline with optional config overrides."
        )
        ingest_parser.add_argument(
            "--signal-input-path",
            type=str,
            help="Path to the signal-enriched input CSV file (overrides config or .env)",
        )
        ingest_parser.add_argument(
            "--config",
            type=str,
            required=False,
            help="Path to the configuration file for ingestion.",
        )
        ingest_parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode during ingestion.",
        )

        # Subcommand: train
        train_parser = subparsers.add_parser(
            "train", help="Train the model using the configured pipeline."
        )
        train_parser.add_argument(
            "--config",
            type=str,
            required=False,
            default="config/model_config.yaml",
            help="Path to the configuration file for training.",
        )
        train_parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode during training.",
        )

        # Parse the arguments
        args = parser.parse_args()

        # Check if a subcommand was provided
        if args.command is None:
            parser.print_help()
            exit(1)

        # Return a CommandLineArgs object with parsed values
        return CommandLineArgs(
            command=args.command,
            config=args.config,
            debug=args.debug,
            signal_input_path=getattr(args, "signal_input_path", None),
        )
