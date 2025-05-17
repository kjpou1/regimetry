# regimetry/services/model_embedding_service.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from regimetry.logger_manager import LoggerManager
from regimetry.config.config import Config


logging = LoggerManager.get_logger(__name__)


class ModelEmbeddingService:
    def __init__(self, input_shape):
        self.config = Config()
        self.input_shape = input_shape
        self.model = self._build_model()

        # Optional safety check
        embedding_dim = self.config.embedding_dim
        if embedding_dim is not None:
            assert self.input_shape[-1] == embedding_dim, (
                f"❌ Mismatch: input dim {self.input_shape[-1]} ≠ embedding_dim {embedding_dim}"
            )

    def _build_model(self):
        """
        Build and return a Transformer encoder model for embedding extraction.
        """
        head_size = self.config.head_size
        num_heads = self.config.num_heads
        ff_dim = self.config.ff_dim
        num_blocks = self.config.num_transformer_blocks
        dropout = self.config.dropout

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
