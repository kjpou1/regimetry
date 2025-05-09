import numpy as np
import tensorflow as tf


class PositionalEncoding:
    @staticmethod
    def get_angles(pos, i, d_model):
        """
        Compute the angle rates for sinusoidal positional encodings.

        Args:
            pos (ndarray): Position indices, shape (seq_len, 1)
            i (ndarray): Dimension indices, shape (1, d_model)
            d_model (int): Embedding dimension

        Returns:
            ndarray: (seq_len, d_model) matrix of angle values
        """
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    @staticmethod
    def get_sinusoidal_encoding(seq_len, d_model):
        """
        Generate sinusoidal positional encodings with safe broadcasting.

        This method avoids broadcast errors by computing the full (seq_len, d_model)
        matrix of angles using `get_angles`, and then slicing it to apply sin and cos.

        Unlike earlier approaches that constructed `div_term` manually (which can lead
        to mismatches when d_model is odd), this ensures that:
        - The full shape is precomputed
        - Slicing for sin and cos respects actual index spacing
        - No shape mismatch occurs even when d_model is odd

        Args:
            seq_len (int): Sequence length
            d_model (int): Embedding dimension

        Returns:
            tf.Tensor: Positional encoding tensor of shape (seq_len, d_model)
        """
        pos = np.arange(seq_len)[:, np.newaxis]       # (seq_len, 1)
        i = np.arange(d_model)[np.newaxis, :]         # (1, d_model)
        angle_rads = PositionalEncoding.get_angles(pos, i, d_model)

        # Apply sin to even indices (0, 2, 4, ...)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices (1, 3, 5, ...)
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return tf.convert_to_tensor(angle_rads, dtype=tf.float32)

    @staticmethod
    def add(inputs, method='sinusoidal', learnable_dim=None):
        """
        Add positional encodings to a 3D input tensor (batch, seq_len, d_model).

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            method (str): 'sinusoidal' or 'learnable'
            learnable_dim (int, optional): Required if method is 'learnable'

        Returns:
            tf.Tensor: Tensor with positional encodings added
        """
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]

        if method == 'sinusoidal':
            pe = PositionalEncoding.get_sinusoidal_encoding(seq_len, d_model)
            return inputs + tf.expand_dims(pe, axis=0)

        elif method == 'learnable':
            if learnable_dim is None:
                raise ValueError("learnable_dim must be set for learnable encoding.")

            seq_len = inputs.shape[1]
            if seq_len is None:
                raise ValueError("Sequence length must be statically defined for learnable encoding.")

            pos_embedding = tf.keras.layers.Embedding(input_dim=seq_len, output_dim=learnable_dim)
            positions = tf.range(seq_len)
            pe = pos_embedding(positions)  # (seq_len, learnable_dim)
            return inputs + tf.expand_dims(pe, axis=0)

        else:
            raise ValueError(f"Unsupported method: {method}")
