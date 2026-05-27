import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant


class MacroF1Monitor(Callback):
    """
    Custom EarlyStopping that monitors validation **macro F1** instead of
    validation loss.

    Why this is needed:
        Keras' `class_weight` parameter re-weights the *training* loss but
        the *validation* loss stays unweighted. On a heavily imbalanced
        validation set, the unweighted val_loss is minimised by simply
        predicting the majority class — so a built-in
        EarlyStopping(monitor='val_loss', restore_best_weights=True) silently
        reverts the model to its degenerate cold-start state. Macro F1 is the
        right signal: it cannot be gamed by predicting one class.
    """

    def __init__(self, X_val, y_val, patience=10, verbose=1):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.patience = patience
        self.verbose = verbose
        self.best_f1 = -1.0
        self.best_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0).argmax(axis=1)
        macro_f1 = f1_score(self.y_val, preds, average='macro', zero_division=0)
        if logs is not None:
            logs['val_macro_f1'] = macro_f1

        if macro_f1 > self.best_f1 + 1e-4:
            self.best_f1 = macro_f1
            self.best_weights = self.model.get_weights()
            self.wait = 0
            if self.verbose:
                print(f"  Epoch {epoch + 1}: val_macro_f1 improved to {macro_f1:.4f}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"  Epoch {epoch + 1}: val_macro_f1 = {macro_f1:.4f}  (no improvement, wait={self.wait}/{self.patience})")
            if self.wait >= self.patience:
                self.model.stop_training = True
                if self.verbose:
                    print(f"  Stopping early — best val_macro_f1 = {self.best_f1:.4f}")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            if self.verbose:
                print(f"  Restored weights from best epoch (val_macro_f1 = {self.best_f1:.4f})")


class DeepLearning:
    """
    Deep Learning model (Multi-Layer Perceptron) for the flight delay problem,
    implemented with TensorFlow / Keras layer-by-layer.

    Two tasks are supported on the same data preparation:
        - 'classification' (default): 3 classes (On-time / Short / Long)
        - 'regression'              : predict ARR_DELAY in minutes

    Architecture (classification head):
        Input(d) -> Dense(128, relu) -> BatchNorm -> Dropout(0.3)
                 -> Dense(64,  relu) -> BatchNorm -> Dropout(0.3)
                 -> Dense(32,  relu) -> Dropout(0.2)
                 -> Dense(3, softmax)

    Why MLP and not CNN / RNN:
        - The data is fully tabular: numeric scaled features + one-hot encoded
          categorical features. There is no spatial structure (CNN) and no
          temporal sequence per sample (RNN/LSTM) — every row is an independent
          flight. The MLP is the canonical deep-learning architecture for this
          kind of structured tabular data and was the same choice taken in the
          reference project and the theoretical-practical classes.

    Attributes:
        data_loader        : DataLoader object holding preprocessed train/test data.
        task               : 'classification' or 'regression'.
        train_sample_size  : Optional cap on training rows (speeds up experiments).
        test_sample_size   : Optional cap on testing rows.
        balance_training   : If True, undersample majority classes (classification only).
        random_state       : Seed for reproducibility.
        model              : Compiled Keras model (after build_model()).
        history            : Keras History object (after train()).
        results            : Dict with the evaluation metrics.

    Methods:
        build_model     : Construct and compile the Keras Sequential MLP.
        train           : Fit the model with EarlyStopping + ReduceLROnPlateau.
        evaluate        : Run on test data and report the appropriate metrics.
        plot_history    : Plot training / validation curves.
        run_all         : Convenience pipeline: build -> train -> evaluate -> plot.
    """

    CLASS_NAMES = ['On-time', 'Short', 'Long']

    def __init__(self, data_loader, task='classification',
                 train_sample_size=None, test_sample_size=None,
                 balance_training=False, random_state=42):
        self.data_loader = data_loader
        self.task = task
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.balance_training = balance_training
        self.random_state = random_state

        # Reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.model = None
        self.history = None
        self.results = {}

        self._prepare_data()

    @staticmethod
    def _categorize_delay(delay):
        if delay < 15:
            return 0
        elif delay <= 30:
            return 1
        else:
            return 2

    def _prepare_data(self):
        # Drop non-numeric columns (same approach used in the KNN / Ensemble sections)
        cols_to_drop = self.data_loader.data_train.select_dtypes(
            exclude=['number', 'bool']
        ).columns
        if len(cols_to_drop) > 0:
            print(f"Dropping non-numeric columns: {list(cols_to_drop)}")

        X_train = self.data_loader.data_train.drop(columns=cols_to_drop).values.astype(np.float32)
        X_test = self.data_loader.data_test.drop(columns=cols_to_drop).values.astype(np.float32)

        if self.task == 'classification':
            y_train = self.data_loader.labels_train.apply(self._categorize_delay).values
            y_test = self.data_loader.labels_test.apply(self._categorize_delay).values
        else:  # regression
            y_train = self.data_loader.labels_train.values.astype(np.float32)
            y_test = self.data_loader.labels_test.values.astype(np.float32)

        # Optional subsampling for fast experimentation
        if self.train_sample_size is not None:
            X_train = X_train[:self.train_sample_size]
            y_train = y_train[:self.train_sample_size]
        if self.test_sample_size is not None:
            X_test = X_test[:self.test_sample_size]
            y_test = y_test[:self.test_sample_size]

        # Class balancing (classification only)
        if self.balance_training and self.task == 'classification':
            rng = np.random.default_rng(self.random_state)
            classes, counts = np.unique(y_train, return_counts=True)
            min_count = counts.min()
            balanced_idx = np.concatenate([
                rng.choice(np.where(y_train == c)[0], size=min_count, replace=False)
                for c in classes
            ])
            rng.shuffle(balanced_idx)
            X_train, y_train = X_train[balanced_idx], y_train[balanced_idx]
            print(
                "Class distribution after balancing: "
                f"{ {self.CLASS_NAMES[c]: int(min_count) for c in classes} }"
            )

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.n_features = X_train.shape[1]

        print(f"Train set: {self.X_train.shape}  |  Test set: {self.X_test.shape}")

    def build_model(self, hidden_units=(128, 64, 32), dropout_rate=0.3,
                    learning_rate=1e-3, use_batch_norm=True):
        """
        Build the MLP layer-by-layer. The architecture is intentionally simple
        but uses BatchNormalization (faster convergence) and Dropout
        (regularization) — the standard recipe for a tabular MLP.

        `use_batch_norm` can be disabled for class-weighted training: BatchNorm
        running statistics are dominated by the majority class, which dampens
        the effect of the per-sample weights in the loss and pushes the model
        toward a degenerate "always predict majority" solution.
        """
        model = Sequential()
        model.add(Input(shape=(self.n_features,)))

        for i, units in enumerate(hidden_units):
            model.add(Dense(units, activation='relu'))
            if use_batch_norm:
                model.add(BatchNormalization())
            # Slightly less dropout on the last hidden layer
            rate = dropout_rate if i < len(hidden_units) - 1 else max(dropout_rate - 0.1, 0.0)
            if rate > 0:
                model.add(Dropout(rate))

        if self.task == 'classification':
            # Initialize output bias to log(class priors). This gives the
            # network a head start that already matches the marginal class
            # distribution, so the optimizer doesn't waste the first epochs
            # collapsing onto the majority class before the class_weight
            # signal can take effect.
            classes, counts = np.unique(self.y_train, return_counts=True)
            priors = counts / counts.sum()
            log_priors = np.log(priors).astype(np.float32)
            model.add(Dense(3, activation='softmax',
                            bias_initializer=Constant(log_priors)))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(Dense(1))  # linear output for regression
            loss = 'mse'
            metrics = ['mae']

        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss=loss, metrics=metrics)

        self.model = model
        print("\n--- Model Architecture ---")
        model.summary()
        return model

    def train(self, epochs=100, batch_size=512, validation_split=0.2, patience=10,
              verbose=1, class_weight_override=None):
        """
        Fit the model with EarlyStopping (restore best weights) and
        ReduceLROnPlateau. For unbalanced classification we automatically pass
        sklearn-computed class_weight so the loss does not collapse onto the
        majority class.

        `class_weight_override` lets the caller pass an explicit dict (e.g.
        {0: 1.0, 1: 5.0, 2: 2.5}) to apply a stronger penalty than the
        sklearn 'balanced' heuristic, which empirically is too mild for our
        70/10/20 class distribution and lets the model collapse to majority.
        """
        if self.model is None:
            self.build_model()

        class_weight = None
        if self.task == 'classification' and not self.balance_training:
            if class_weight_override is not None:
                class_weight = class_weight_override
                print(f"Using OVERRIDE class_weight = {class_weight}")
            else:
                classes = np.unique(self.y_train)
                weights = compute_class_weight(class_weight='balanced',
                                               classes=classes, y=self.y_train)
                class_weight = dict(zip(classes, weights))
                print(f"Using class_weight = {class_weight}")

        # Manual stratified split so we can pass an explicit validation_data
        # to model.fit() AND see the same X_val/y_val from the MacroF1 callback.
        stratify = self.y_train if self.task == 'classification' else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=validation_split,
            stratify=stratify,
            random_state=self.random_state,
        )

        if self.task == 'classification':
            # Critical: monitor macro F1 (immune to majority collapse) instead
            # of the unweighted val_loss that built-in EarlyStopping would use.
            callbacks = [
                MacroF1Monitor(X_val, y_val, patience=patience, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=max(patience // 2, 3), verbose=1),
            ]
        else:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=patience,
                              restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=max(patience // 2, 3), verbose=1),
            ]

        print(f"\nTraining for up to {epochs} epochs (batch_size={batch_size})...")
        self.history = self.model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=verbose,
        )
        return self.history

    def evaluate(self):
        """Run on the held-out test set and report task-appropriate metrics."""
        if self.task == 'classification':
            probs = self.model.predict(self.X_test, verbose=0)
            preds = np.argmax(probs, axis=1)

            acc = accuracy_score(self.y_test, preds)
            report = classification_report(
                self.y_test, preds,
                target_names=self.CLASS_NAMES, zero_division=0,
            )
            cm = confusion_matrix(self.y_test, preds)
            f1_per_class = f1_score(self.y_test, preds, average=None, zero_division=0)
            macro_f1 = f1_score(self.y_test, preds, average='macro', zero_division=0)

            self.results = {
                'accuracy': acc,
                'report': report,
                'cm': cm,
                'f1_ontime': f1_per_class[0],
                'f1_short': f1_per_class[1],
                'f1_long': f1_per_class[2],
                'macro_f1': macro_f1,
            }

            print("\n--- MLP (Deep Learning) ---")
            print(f"Accuracy: {acc * 100:.2f}%")
            print(report)

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.CLASS_NAMES, yticklabels=self.CLASS_NAMES)
            plt.title('Confusion Matrix - MLP (Deep Learning)')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.show()
        else:
            preds = self.model.predict(self.X_test, verbose=0).flatten()
            mse = mean_squared_error(self.y_test, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, preds)
            r2 = r2_score(self.y_test, preds)

            self.results = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

            print("\n--- MLP Regression (Deep Learning) ---")
            print(f"MSE  : {mse:.3f}")
            print(f"RMSE : {rmse:.3f}  minutes")
            print(f"MAE  : {mae:.3f}  minutes")
            print(f"R^2  : {r2:.4f}")

            # Predicted vs actual scatter (sample to keep plot light)
            n_plot = min(2000, len(preds))
            idx = np.random.choice(len(preds), n_plot, replace=False)
            plt.figure(figsize=(6, 6))
            plt.scatter(self.y_test[idx], preds[idx], alpha=0.3, s=10)
            lo, hi = self.y_test[idx].min(), self.y_test[idx].max()
            plt.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Ideal')
            plt.xlabel('Actual ARR_DELAY (min)')
            plt.ylabel('Predicted ARR_DELAY (min)')
            plt.title('MLP Regression: Predicted vs Actual')
            plt.legend()
            plt.tight_layout()
            plt.show()

        return self.results

    def plot_history(self):
        """Plot the training / validation curves to inspect under/overfitting."""
        if self.history is None:
            print("No training history available — call train() first.")
            return

        h = self.history.history
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # Loss
        axes[0].plot(h['loss'], label='Train Loss', marker='o', markersize=3)
        axes[0].plot(h['val_loss'], label='Val Loss', marker='o', markersize=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss curve')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Secondary metric: prefer macro_f1 when our custom callback produced it.
        if self.task == 'classification' and 'val_macro_f1' in h:
            axes[1].plot(h['val_macro_f1'], label='Val Macro F1',
                         marker='o', markersize=3, color='tab:green')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Macro F1')
            axes[1].set_title('Validation Macro F1')
        else:
            metric_key = 'accuracy' if self.task == 'classification' else 'mae'
            if metric_key in h:
                axes[1].plot(h[metric_key], label=f'Train {metric_key}', marker='o', markersize=3)
                axes[1].plot(h[f'val_{metric_key}'], label=f'Val {metric_key}', marker='o', markersize=3)
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel(metric_key)
                axes[1].set_title(f'{metric_key.capitalize()} curve')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_all(self, hidden_units=(128, 64, 32), dropout_rate=0.3,
                learning_rate=1e-3, epochs=100, batch_size=512,
                patience=10, verbose=1, use_batch_norm=True,
                class_weight_override=None):
        """Convenience pipeline: build -> train -> plot history -> evaluate."""
        self.build_model(hidden_units=hidden_units,
                         dropout_rate=dropout_rate,
                         learning_rate=learning_rate,
                         use_batch_norm=use_batch_norm)
        self.train(epochs=epochs, batch_size=batch_size,
                   patience=patience, verbose=verbose,
                   class_weight_override=class_weight_override)
        self.plot_history()
        self.evaluate()
        return self.results
