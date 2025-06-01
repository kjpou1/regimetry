# src/regimetry/config/training_profile_config.py

import os
from dataclasses import dataclass, field

import yaml
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from regimetry.config.config import Config


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
        model_checkpoint=None,
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

        self.model_checkpoint = model_checkpoint or {
            "enabled": False,
            "mode": "min",
            "save_best_only": True,
            "save_weights_only": False,
            "filename": "best_model.keras",
            "verbose": 0,
        }

        self.checkpoint_enabled = self.model_checkpoint.get("enabled", False)
        self.checkpoint_mode = self.model_checkpoint.get("mode", "min")
        self.checkpoint_save_best_only = self.model_checkpoint.get(
            "save_best_only", True
        )
        self.checkpoint_save_weights_only = self.model_checkpoint.get(
            "save_weights_only", False
        )
        self.checkpoint_filename = self.model_checkpoint.get(
            "filename", "best_model.keras"
        )
        self.checkpoint_verbose = self.model_checkpoint.get("verbose", 0)

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

    def get_callbacks(self, output_dir: str = None):
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

        if self.checkpoint_enabled:
            config = Config()
            best_model_path = config._resolve_path(
                os.path.join(config.output_dir, self.checkpoint_filename)
            )
            callbacks.append(
                ModelCheckpoint(
                    filepath=best_model_path,
                    monitor=monitor_metric,
                    mode=self.checkpoint_mode,
                    save_best_only=self.checkpoint_save_best_only,
                    save_weights_only=self.checkpoint_save_weights_only,
                    verbose=self.checkpoint_verbose,
                )
            )

        return callbacks
