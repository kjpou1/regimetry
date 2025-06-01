from dataclasses import dataclass


@dataclass
class ModelCheckpointConfig:
    enabled: bool = False
    monitor: str = "val_loss"
    mode: str = "min"
    save_best_only: bool = True
    save_weights_only: bool = False
    filename: str = "best_model.keras"
