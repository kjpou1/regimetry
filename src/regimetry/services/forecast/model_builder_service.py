import os

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras.optimizers import Adam

from regimetry.config.config import Config
from regimetry.config.training_profile_config import TrainingProfileConfig
from regimetry.logger_manager import LoggerManager
from regimetry.models.forecast.forecaster_factory import ForecasterFactory
from regimetry.utils.loss_utils import get_loss_function

logging = LoggerManager.get_logger(__name__)


class ForecastModelBuilderService:
    """
    ForecastModelBuilderService

    Responsible for building and compiling a forecasting model using
    the specified architecture and configuration. Supports normalization,
    loss setup, optimizer, and callbacks.
    """

    def __init__(self, training_profile=None):
        self.config = Config()
        # If no training profile is passed, load from config's path or default YAML
        if training_profile is not None:
            self.training_profile = training_profile
        else:
            profile_path = self.config.training_profile_path
            if profile_path is not None:
                self.training_profile = TrainingProfileConfig.from_yaml(profile_path)
            else:
                # Hard fallback: default inline config
                self.training_profile = TrainingProfileConfig()

        # Optional: sanity log
        print(
            f"[ForecastModelBuilderService] Using training profile: {self.training_profile.model_type}"
        )

    def build_model(
        self, input_shape: tuple, output_dim: int, normalize_output: bool = True
    ):
        """
        Builds and compiles the forecast model.

        Args:
            input_shape (tuple): Shape of input windows (timesteps, features)
            output_dim (int): Size of the output embedding vector

        Returns:
            model (tf.keras.Model): Compiled Keras model ready for training
            callbacks (list): List of Keras callbacks to use during training
        """
        model_type = self.training_profile.model_type

        logging.info(f"🛠️ Building model: {model_type} (normalize={normalize_output})")

        # Initialize architecture factory
        factory = ForecasterFactory(input_shape, output_dim, normalize_output)

        # Build the model
        model = factory.build(model_type=model_type)

        # Compile the model
        # model.compile(
        #     optimizer=Adam(learning_rate=0.001),
        #     loss=cosine_similarity,
        # )
        model.compile(
            optimizer=self.training_profile.get_optimizer(),
            loss=get_loss_function(self.training_profile.loss),
        )
        # Set up callbacks
        callbacks = self.training_profile.get_callbacks()

        return model, callbacks
