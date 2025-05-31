import tensorflow.keras.backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Lambda,
    LayerNormalization,
    MultiHeadAttention,
)


class ForecasterFactory:
    """
    A flexible builder for LSTM-based forecasting architectures.

    Supported model types:
        - 'simple': Single-layer LSTM
        - 'stratum': Stacked LSTM + dropout
        - 'stratum_bi': BiLSTM + LSTM + dropout
        - 'stratum_attn': BiLSTM + self-attention (functional)
        - 'stratum_hydra': BiLSTM + MultiHeadAttention + pooling

    Parameters:
        input_shape (tuple): Input shape of the time series (timesteps, features)
        output_dim (int): Output dimensionality of the forecast embedding
        normalize_output (bool): Whether to apply L2 normalization to output
    """

    def __init__(self, input_shape, output_dim, normalize_output=True):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.normalize_output = normalize_output

    def build(self, model_type="simple"):
        if model_type == "stratum_hydra":
            return self._build_stratum_hydra()

        elif model_type == "stratum_attn":
            return self._build_stratum_attn()

        else:
            return self._build_sequential(model_type)

    def _build_stratum_hydra(self):
        inputs = Input(shape=self.input_shape, name="input")
        x = LSTM(64, return_sequences=True)(inputs)
        x = LayerNormalization()(x)
        x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(self.output_dim)(x)

        if self.normalize_output:
            x = Lambda(lambda t: K.l2_normalize(t, axis=-1))(x)

        return Model(inputs, x, name="stratum_hydra_forecaster")

    def _build_stratum_attn(self):
        inputs = Input(shape=self.input_shape, name="input")
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = Dropout(0.2)(x)
        x = LayerNormalization()(x)

        # Attention mechanism: score → weights → context
        score = Dense(1, activation="tanh")(x)
        weights = Lambda(lambda s: K.softmax(s, axis=1))(score)
        context = Lambda(lambda z: K.sum(z[0] * z[1], axis=1))([x, weights])

        x = Dense(self.output_dim)(context)
        if self.normalize_output:
            x = Lambda(lambda t: K.l2_normalize(t, axis=-1))(x)

        return Model(inputs, x, name="stratum_attn_forecaster")

    def _build_sequential(self, model_type):
        model = Sequential(name=f"{model_type}_forecaster")

        if model_type == "simple":
            model.add(LSTM(64, input_shape=self.input_shape))

        elif model_type == "stratum":
            model.add(LSTM(64, return_sequences=True, input_shape=self.input_shape))
            model.add(LSTM(64))
            model.add(Dropout(0.2))

        elif model_type == "stratum_bi":
            model.add(
                Bidirectional(
                    LSTM(64, return_sequences=True), input_shape=self.input_shape
                )
            )
            model.add(LSTM(64))
            model.add(Dropout(0.2))

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.add(Dense(self.output_dim))
        if self.normalize_output:
            model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))

        return model
