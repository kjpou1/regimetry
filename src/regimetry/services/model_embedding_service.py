# regimetry/services/model_embedding_service.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from regimetry.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class ModelEmbeddingService:
    def __init__(self, input_shape, config=None):
        self.input_shape = input_shape
        self.config = config or {}
        self.model = self._build_model()

    def _build_model(self):
        """
        Build and return a Transformer encoder model for embedding extraction.
        """
        head_size = self.config.get("head_size", 256)
        num_heads = self.config.get("num_heads", 4)
        ff_dim = self.config.get("ff_dim", 128)
        num_blocks = self.config.get("num_transformer_blocks", 2)
        dropout = self.config.get("dropout", 0.1)

        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        for _ in range(num_blocks):
            x = self._transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D()(x)
        return keras.Model(inputs=inputs, outputs=x, name="UnsupervisedTransformerEncoder")

    def _transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout):
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def embed(self, X_pe, batch_size=64):
        """
        Generate transformer embeddings from positionally encoded input.
        """
        logging.info(f"⚙️ Generating embeddings: input shape = {X_pe.shape}")
        embeddings = self.model.predict(X_pe, batch_size=batch_size)
        logging.info(f"✅ Embeddings shape: {embeddings.shape}")
        return embeddings
