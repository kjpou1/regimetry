import asyncio
import os

from regimetry.config.config import Config
from regimetry.config.dynamic_config_loader import DynamicConfigLoader
from regimetry.exception import CustomException
from regimetry.logger_manager import LoggerManager
from regimetry.models.command_line_args import CommandLineArgs
from regimetry.pipelines.embedding_pipeline import EmbeddingPipeline
from regimetry.pipelines.forecast_trainer_pipeline import ForecastTrainerPipeline
from regimetry.pipelines.ingestion_pipeline import IngestionPipeline
from regimetry.pipelines.regime_clustering_pipeline import RegimeClusteringPipeline
from regimetry.pipelines.regime_interpretability_pipeline import run as interpret_run

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
            logging.info(
                f"Overriding signal_input_path from CLI: {args.signal_input_path}"
            )
            self.config.signal_input_path = args.signal_input_path

        if args.output_name:
            logging.info(f"Overriding output_name from CLI: {args.output_name}")
            self.config.output_name = args.output_name

        if args.window_size:
            logging.info(f"Overriding window_size from CLI: {args.window_size}")
            self.config.window_size = args.window_size

        if args.stride:
            logging.info(f"Overriding stride from CLI: {args.stride}")
            self.config.stride = args.stride

        if args.encoding_method:
            logging.info(f"Overriding encoding_method from CLI: {args.encoding_method}")
            self.config.encoding_method = args.encoding_method

        if args.encoding_style:
            logging.info(f"Overriding encoding_style from CLI: {args.encoding_style}")
            self.config.encoding_style = args.encoding_style

        if args.embedding_dim is not None:
            logging.info(f"Overriding embedding_dim from CLI: {args.embedding_dim}")
            self.config.embedding_dim = args.embedding_dim

        if args.forecast_embedding_dir:
            logging.info(
                f"Overriding embedding_dir from CLI: {args.forecast_embedding_dir}"
            )
            self.config.embedding_dir = args.forecast_embedding_dir

        if args.forecast_cluster_assignment_path:
            logging.info(
                f"Overriding cluster_assignment_path from CLI: {args.forecast_cluster_assignment_path}"
            )
            self.config.cluster_assignment_path = args.forecast_cluster_assignment_path

        if args.forecast_model_type:
            logging.info(f"Overriding model_type from CLI: {args.forecast_model_type}")
            self.config.model_type = args.forecast_model_type

        if args.forecast_n_neighbors is not None:
            logging.info(
                f"Overriding n_neighbors from CLI: {args.forecast_n_neighbors}"
            )
            self.config.n_neighbors = args.forecast_n_neighbors

        if args.instrument:
            logging.info(f"Overriding instrument from CLI: {args.instrument}")
            self.config.instrument = args.instrument

        if args.base_config:
            logging.info(f"Overriding base_config from CLI: {args.base_config}")
            self.config.base_config = args.base_config

        if self.args.output_dir:
            logging.info(f"Overriding output_dir from CLI: {self.args.output_dir}")
            self.config.output_dir = self.args.output_dir

        if self.args.command == "forecast" and self.args.forecast_command == "train":
            if args.profile_path:
                logging.info(
                    f"Overriding training_profile_path from CLI: {args.profile_path}"
                )
                self.config.training_profile_path = args.profile_path

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
            elif self.args.command == "interpret":
                logging.info("Executing interpretability workflow.")
                await self.run_interpret()
            elif self.args.command == "analyze":
                logging.info("Executing dynamic analyze pipeline.")
                await self.run_analyze()
            elif self.args.command == "forecast":
                if self.args.forecast_command == "train":
                    logging.info("Executing forecast training pipeline.")
                    await self.run_forecast_train()
                else:
                    raise ValueError(
                        "Please specify a valid subcommand under 'forecast': 'train'"
                    )

            else:
                logging.error("No valid subcommand provided.")
                raise ValueError(
                    "Please specify a valid subcommand: 'ingest', 'embed', or 'cluster'."
                )

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
        pipeline = IngestionPipeline(val_size=0.2, test_size=0.2)

        result = pipeline.run()

        logging.info("Ingestion pipeline result:")
        logging.info(f"  Full Dataset: {result['output_path']}")
        logging.info(f"  Features: {result['features']}")

    async def run_embedding(self):
        """
        Execute the model training workflow by calling the model training service.
        """
        try:
            embed_pipeline = EmbeddingPipeline()
            embed_pipeline.run_pipeline()

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
                logging.info(
                    f"Overriding embedding_path from CLI: {self.args.embedding_path}"
                )
                self.config.embedding_path = self.args.embedding_path

            if self.args.regime_data_path:
                logging.info(
                    f"Overriding regime_data_path from CLI: {self.args.regime_data_path}"
                )
                self.config.regime_data_path = self.args.regime_data_path

            if self.args.window_size:
                logging.info(
                    f"Overriding window_size from CLI: {self.args.window_size}"
                )
                self.config.window_size = self.args.window_size

            if self.args.n_clusters:
                logging.info(f"Overriding n_clusters from CLI: {self.args.n_clusters}")
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
                raise ValueError(
                    f"Missing required config fields: {', '.join(missing)}"
                )

            pipeline = RegimeClusteringPipeline()
            pipeline.run()

        except Exception as e:
            logging.error(f"❌ Error during clustering: {e}")
            raise e

    async def run_interpret(self):
        """
        Run the regime interpretability pipeline.
        """
        logging.info("Executing regime interpretability pipeline.")

        if not self.args.input_path:
            raise ValueError("❌ --input-path is required for interpret command.")
        if not self.args.output_dir:
            raise ValueError("❌ --output-dir is required for interpret command.")

        interpret_run(
            input_path=self.args.input_path,
            output_dir=self.args.output_dir,
            cluster_col=self.args.cluster_col or "Cluster_ID",
            save_csv=self.args.save_csv,
            save_heatmap=self.args.save_heatmap,
            save_json=self.args.save_json,
        )

    async def run_analyze(self):
        """
        Dynamically run embedding and clustering pipelines based on runtime metadata.
        Automatically skips steps if output already exists.
        """
        logging.info("🔍 Starting dynamic analyze pipeline")

        loader = DynamicConfigLoader()

        config = loader.load(
            instrument=self.args.instrument,
            window_size=self.args.window_size,
            stride=self.args.stride,
            encoding_method=self.args.encoding_method,
            encoding_style=self.args.encoding_style,
            embedding_dim=self.args.embedding_dim,
            n_clusters=self.args.n_clusters,
            export_path=os.path.join(self.config.BASE_DIR, "tmp_config.yaml"),
            create_dirs=self.args.create_dir,
            force=self.args.force,
            clean=self.args.clean,
        )

        # Sync config object with injected paths
        self.config.load_from_yaml("artifacts/tmp_config.yaml")

        embedding_path = self.config.embedding_path
        cluster_path = config["cluster_output_path"]

        if self.args.force or not os.path.exists(embedding_path):
            logging.info("🧠 Embedding not found. Running embedding pipeline.")
            await self.run_embedding()
        else:
            logging.info("✅ Embedding already exists, skipping embedding step.")

        if self.args.force or not os.path.exists(cluster_path):
            logging.info(
                "🔗 Cluster assignments not found. Running clustering pipeline."
            )
            await self.run_clustering()
        else:
            logging.info("✅ Cluster file already exists, skipping clustering step.")

        # Optionally add automatic visualization/interpret here
        logging.info("🧾 Analyze pipeline complete. Outputs located in:")
        logging.info(f"   Embedding: {embedding_path}")
        logging.info(f"   Clusters : {cluster_path}")

    async def run_forecast_train(self):
        """
        Run the forecast training pipeline to predict next-cluster embedding and classification.
        """
        forecast_pipeline = ForecastTrainerPipeline()
        forecast_pipeline.run()
