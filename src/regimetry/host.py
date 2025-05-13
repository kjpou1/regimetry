import asyncio
import os

from regimetry.config.config import Config
from regimetry.exception import CustomException
from regimetry.logger_manager import LoggerManager
from regimetry.models.command_line_args import CommandLineArgs
from regimetry.pipelines.ingestion_pipeline import IngestionPipeline
from regimetry.pipelines.embedding_pipeline import EmbeddingPipeline
from regimetry.pipelines.regime_clustering_pipeline import RegimeClusteringPipeline


logging = LoggerManager.get_logger(__name__)


class Host:
    """
    Host class to manage the execution of the main application.

    This class handles initialization with command-line arguments and
    configuration, and runs the main asynchronous functionality.
    """

    def __init__(self, args: CommandLineArgs):
        """
        Initialize the Host class with command-line arguments and configuration.

        Parameters:
        args (CommandLineArgs): Command-line arguments passed to the script.
        """
        self.args = args
        self.config = Config()
        if args.config:
            self.config.config_path = args.config
            self.config.load_from_yaml(args.config)
            
        if args.signal_input_path:
            logging.info(f"Overriding signal_input_path from CLI: {args.signal_input_path}")
            self.config.signal_input_path = args.signal_input_path

        if args.output_name:
            logging.info(f"Overriding output_name from CLI: {args.output_name}")
            self.config.output_name = args.output_name            

    def run(self):
        """
        Synchronously run the asynchronous run_async method.

        This is a blocking call that wraps the asynchronous method.
        """
        return asyncio.run(self.run_async())

    async def run_async(self):
        """
        Main asynchronous method to execute the host functionality.

        Determines the action based on the provided subcommand.
        """
        try:
            logging.info("Starting host operations.")

            if self.args.command == "ingest":
                logging.info("Executing data ingestion workflow.")
                await self.run_ingestion()
            elif self.args.command == "embed":
                logging.info("Executing embedding workflow.")
                # if not self.args.model_type:
                #     raise ValueError(
                #         "A model type must be specified for the 'train' command."
                #     )
                await self.run_embedding()

            elif self.args.command == "cluster":
                logging.info("Executing clustering workflow.")
                await self.run_clustering()
            else:
                logging.error("No valid subcommand provided.")
                raise ValueError("Please specify a valid subcommand: 'ingest', 'embed', or 'cluster'.")

        except CustomException as e:
            logging.error("A custom error occurred during host operations: %s", e)
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise  # CustomException("An unexpected error occured", sys) from e
        finally:
            logging.info("Shutting down host gracefully.")

    async def run_ingestion(self):
        """
        Run the supervised ingestion pipeline from CSV input to processed train/val/test outputs.
        """
        logging.info("Initializing ingestion pipeline...")

        # You can optionally parameterize these via CLI args later
        pipeline = IngestionPipeline(
            val_size=0.2,
            test_size=0.2
        )

        result = pipeline.run()

        logging.info("Ingestion pipeline result:")
        logging.info(f"  Full Dataset: {result['output_path']}")
        logging.info(f"  Features: {result['features']}")


    async def run_embedding(self):
        """
        Execute the model training workflow by calling the model training service.
        """
        try:
            train_pipeline = EmbeddingPipeline()
            train_pipeline.run_pipeline()

        except Exception as e:
            logging.error(f"Error during host training: {e}")
            raise e

    async def run_clustering(self):
        """
        Run the unsupervised clustering pipeline using transformer embeddings.
        Loads parameters from config or CLI overrides.
        """
        try:
            # CLI > config.yaml precedence
            if self.args.embedding_path:
                self.config.embedding_path = self.args.embedding_path
            if self.args.regime_data_path:
                self.config.regime_data_path = self.args.regime_data_path
            if self.args.output_dir:
                self.config.output_dir = self.args.output_dir
            if self.args.window_size:
                self.config.window_size = self.args.window_size
            if self.args.n_clusters:
                self.config.n_clusters = self.args.n_clusters

            # Manual validation of required fields
            missing = []
            if not self.config.embedding_path:
                missing.append("embedding_path")
            if not self.config.regime_data_path:
                missing.append("regime_data_path")
            if not self.config.output_dir:
                missing.append("output_dir")
            if missing:
                raise ValueError(f"Missing required config fields: {', '.join(missing)}")

            pipeline = RegimeClusteringPipeline()
            pipeline.run()

        except Exception as e:
            logging.error(f"❌ Error during clustering: {e}")
            raise e
