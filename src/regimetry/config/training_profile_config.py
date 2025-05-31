# src/regimetry/config/training_profile_config.py

from dataclasses import dataclass, field

import yaml
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam, RMSprop


@dataclass
class TrainingProfileConfig:

    def __init__(
        self,
        model_type="stratum",
        loss="mse",
        optimizer=None,
        epochs=100,
        batch_size=64,
        use_validation=False,
        early_stopping=None,
        lr_scheduler=None,
        normalize_output=True,
        validation_split=0.2,
        verbose=1,
        description=None,
    ):
        self.model_type = model_type
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_validation = use_validation
        self.normalize_output = normalize_output
        self.validation_split = validation_split
        self.verbose = verbose
        self.description = description

        self.optimizer_config = optimizer or {"type": "adam", "learning_rate": 0.001}
        self.optimizer_type = self.optimizer_config.get("type", "adam").lower()
        self.learning_rate = self.optimizer_config.get("learning_rate", 0.001)

        self.early_stopping = early_stopping or {
            "enabled": True,
            "patience": 10,
            "restore_best_weights": True,
        }

        self.early_stopping_enabled = self.early_stopping.get("enabled", True)
        self.early_stopping_patience = self._parse_int(
            self.early_stopping.get("patience"), 10
        )
        self.early_stopping_restore_best_weights = self.early_stopping.get(
            "restore_best_weights", True
        )
        self.early_stopping_verbose = self._parse_int(
            self.early_stopping.get("verbose"), 0
        )

        self.lr_scheduler = lr_scheduler or {
            "enabled": True,
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6,
            "verbose": 1,
        }

        self.lr_scheduler_enabled = self.lr_scheduler.get("enabled", True)
        self.lr_factor = self._parse_float(self.lr_scheduler.get("factor"), 0.5)
        self.lr_patience = self._parse_int(self.lr_scheduler.get("patience"), 5)
        self.lr_min_lr = self._parse_float(self.lr_scheduler.get("min_lr"), 1e-6)
        self.lr_verbose = self._parse_int(self.lr_scheduler.get("verbose"), 0)

    def get_optimizer(self):
        if self.optimizer_type == "adam":
            return Adam(learning_rate=self.learning_rate)
        elif self.optimizer_type == "sgd":
            return SGD(learning_rate=self.learning_rate)
        elif self.optimizer_type == "rmsprop":
            return RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"❌ Unsupported optimizer type: {self.optimizer_type}")

    def _parse_float(self, value, default):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _parse_int(self, value, default):
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @classmethod
    def from_yaml(cls, path: str):
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def get_callbacks(self):
        """Build Keras callbacks based on training settings."""
        monitor_metric = "val_loss" if self.use_validation else "loss"
        callbacks = []

        if self.lr_scheduler_enabled:
            callbacks.append(
                ReduceLROnPlateau(
                    monitor=monitor_metric,
                    factor=self.lr_factor,
                    patience=self.lr_patience,
                    min_lr=self.lr_min_lr,
                    verbose=self.lr_verbose,
                )
            )

        if self.use_validation and self.early_stopping_enabled:
            callbacks.append(
                EarlyStopping(
                    monitor=monitor_metric,
                    patience=self.early_stopping_patience,
                    restore_best_weights=self.early_stopping_restore_best_weights,
                    verbose=self.early_stopping_verbose,
                )
            )
        return callbacks
