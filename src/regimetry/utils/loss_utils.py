import tensorflow.keras.backend as K
from tensorflow.keras.losses import (
    CosineSimilarity,
    MeanAbsoluteError,
    MeanSquaredError,
)
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable()
def cosine_loss(y_true, y_pred):
    """Custom cosine loss: 1 - cosine similarity"""
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return 1 - K.sum(y_true * y_pred, axis=-1)


@register_keras_serializable()
def hybrid_loss(y_true, y_pred):
    """Hybrid of MSE and cosine loss"""
    mse = MeanSquaredError()
    return 0.7 * mse(y_true, y_pred) + 0.3 * cosine_loss(y_true, y_pred)


def get_loss_function(loss_name: str):
    """Resolve loss function from string name."""
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return MeanSquaredError()
    elif loss_name == "mae":
        return MeanAbsoluteError()
    elif loss_name == "cosine_similarity":
        return CosineSimilarity()
    elif loss_name == "cosine_loss":
        return cosine_loss
    elif loss_name == "hybrid_loss":
        return hybrid_loss
    else:
        raise ValueError(f"❌ Unsupported loss function: {loss_name}")
