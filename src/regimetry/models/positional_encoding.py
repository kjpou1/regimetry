import numpy as np
import tensorflow as tf
from regimetry.utils.pad_utils import pad_to_even_dim

class PositionalEncoding:
    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    @staticmethod
    def get_sinusoidal_encoding(seq_len, d_model, encoding_style='interleaved'):
        """
        Generate sinusoidal positional encodings.

        Args:
            seq_len (int): Sequence length
            d_model (int): Embedding dimension
            encoding_style (str): 'interleaved' (Vaswani) or 'stacked' (TensorFlow tutorial)

        Returns:
            tf.Tensor: Positional encoding tensor of shape (seq_len, d_model)
        """
        pos = np.arange(seq_len)[:, np.newaxis]       # (seq_len, 1)
        i = np.arange(d_model)[np.newaxis, :]         # (1, d_model)
        angle_rads = PositionalEncoding.get_angles(pos, i, d_model)

        if encoding_style == 'interleaved':
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        elif encoding_style == 'stacked':
            if d_model % 2 != 0:
                raise ValueError("Stacked style requires even d_model.")
            half_d = d_model // 2
            stacked = np.zeros_like(angle_rads)
            stacked[:, :half_d] = np.sin(angle_rads[:, :half_d])
            stacked[:, half_d:] = np.cos(angle_rads[:, :half_d])
            angle_rads = stacked

        else:
            raise ValueError(f"Unsupported encoding_style: {encoding_style}")

        return tf.convert_to_tensor(angle_rads, dtype=tf.float32)

    @staticmethod
    def add(inputs, method='sinusoidal', learnable_dim=None, encoding_style='interleaved'):
        """
        Add positional encodings to a 3D input tensor (batch, seq_len, d_model).

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            method (str): 'sinusoidal' or 'learnable'
            learnable_dim (int, optional): Required if method is 'learnable'
            encoding_style (str): 'interleaved' or 'stacked' (only used for sinusoidal)

        Returns:
            tf.Tensor: Tensor with positional encodings added
        """
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]

        if method == 'sinusoidal':
            if encoding_style == 'stacked':
                original_d_model = tf.shape(inputs)[-1]
                if original_d_model % 2 != 0:
                    tf.print("[PositionalEncoding] ⚠️ Warning: Input d_model is odd (", original_d_model,
                            "). Padding to even for 'stacked' positional encoding.")
                inputs = pad_to_even_dim(inputs)
                d_model = tf.shape(inputs)[2]
            pe = PositionalEncoding.get_sinusoidal_encoding(seq_len, d_model, encoding_style)
            return inputs + tf.expand_dims(pe, axis=0)

        elif method == 'learnable':
            if learnable_dim is None:
                raise ValueError("learnable_dim must be set for learnable encoding.")
            seq_len = inputs.shape[1]
            if seq_len is None:
                raise ValueError("Sequence length must be statically defined for learnable encoding.")
            pos_embedding = tf.keras.layers.Embedding(input_dim=seq_len, output_dim=learnable_dim)
            positions = tf.range(seq_len)
            pe = pos_embedding(positions)
            return inputs + tf.expand_dims(pe, axis=0)

        else:
            raise ValueError(f"Unsupported method: {method}")
