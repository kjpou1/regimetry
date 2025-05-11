import tensorflow as tf

def pad_to_even_dim(tensor: tf.Tensor) -> tf.Tensor:
    """
    Pads the last dimension of a tensor with 1 extra unit if it is odd.

    Args:
        tensor (tf.Tensor): Tensor of shape (..., d_model)

    Returns:
        tf.Tensor: Tensor with last dim padded to even size (if needed)
    """
    d_model = tf.shape(tensor)[-1]
    pad_width = tf.where(d_model % 2 == 0, 0, 1)
    return tf.pad(tensor, [[0, 0]] * (len(tensor.shape) - 1) + [[0, pad_width]])
