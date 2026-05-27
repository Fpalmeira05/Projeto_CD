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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant

# Reuse the MacroF1Monitor from the flat MLP — it works for binary as well,
# because Macro F1 on a 2-output softmax is exactly the average of per-class F1.
from DeepLearning.DeepLearning import MacroF1Monitor


class HierarchicalDeepLearning:
    """
    Two-stage hierarchical MLP (Flavour B: Classify then Regress).

    Stage 1 — Binary classifier:
        Target: 0 = On-time (ARR_DELAY < 15), 1 = Delayed (ARR_DELAY >= 15).
        Architecture: Dense MLP with Dense(2, softmax) head so the existing
        MacroF1Monitor callback can be reused without modification.
        Trained on the full training set with `class_weight='balanced'`.

    Stage 2 — Regression on minute-level delay:
        Trained on the SUBSET of training flights with ARR_DELAY >= 15.
        Architecture: same MLP backbone with a single linear output unit.
        Standard EarlyStopping(val_loss) — appropriate for regression.

    Inference pipeline:
        For each test flight:
            p_delayed = Stage 1 softmax probability of class 'Delayed'
            if p_delayed >= threshold:
                predicted_minutes = Stage 2 output
            else:
                predicted_minutes = `ontime_minute_value` (default 0)

    Composite evaluation:
        - Per-stage metrics:
            * Stage 1: accuracy, F1 on the 'Delayed' class, Macro F1, confusion.
            * Stage 2: MAE / RMSE / R^2 on the TRUE delayed subset of the test set.
        - End-to-end (chained) metrics:
            * 3-class accuracy and Macro F1, derived by binning the predicted
              minutes (predicted On-time -> class 0; predicted Delayed with
              minutes < 30 -> class 1; >= 30 -> class 2). Directly comparable
              with the flat MLP / Ensemble experiments.
            * End-to-end MAE / RMSE on the full test set's ARR_DELAY (in min).

    Why this beats the flat regression head:
        The flat regression on ARR_DELAY produced R^2 = -1.20 because the
        target spans 0 - 500+ minutes while ~73% of the test set is near zero;
        a single regressor cannot fit both regimes. Filtering Stage 2 to
        delayed flights only (target >= 15) gives the regressor a much tighter
        distribution to fit, while Stage 1 absorbs the easy 'predict 0'
        decisions.

    Attributes:
        data_loader, train_sample_size, test_sample_size, delay_threshold,
        random_state - same conventions as DeepLearning / Ensemble / Knn.
        stage1_model, stage2_model - the trained Keras models.
        stage1_history, stage2_history - Keras History objects after train().
        results - dict with keys 'stage1', 'stage2', 'composite'.

    Methods:
        build_stage1 / build_stage2     - construct each MLP.
        train_stage1 / train_stage2     - fit each stage.
        evaluate_stage1                 - binary metrics + CM.
        evaluate_stage2                 - regression metrics on delayed-only.
        evaluate_composite              - end-to-end cascaded evaluation.
        plot_histories                  - side-by-side training curves.
        run_all                         - one-call convenience pipeline.
    """

    CLASS_NAMES_3 = ['On-time', 'Short', 'Long']
    CLASS_NAMES_2 = ['On-time', 'Delayed']

    def __init__(self, data_loader, train_sample_size=None, test_sample_size=None,
                 delay_threshold=15, random_state=42):
        self.data_loader = data_loader
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.delay_threshold = delay_threshold
        self.random_state = random_state

        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.stage1_model = None
        self.stage2_model = None
        self.stage1_history = None
        self.stage2_history = None
        self.results = {}

        self._prepare_data()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _prepare_data(self):
        cols_to_drop = self.data_loader.data_train.select_dtypes(
            exclude=['number', 'bool']
        ).columns
        if len(cols_to_drop) > 0:
            print(f"Dropping non-numeric columns: {list(cols_to_drop)}")

        X_train = self.data_loader.data_train.drop(columns=cols_to_drop).values.astype(np.float32)
        X_test  = self.data_loader.data_test.drop(columns=cols_to_drop).values.astype(np.float32)

        y_train_min = self.data_loader.labels_train.values.astype(np.float32)
        y_test_min  = self.data_loader.labels_test.values.astype(np.float32)

        if self.train_sample_size is not None:
            X_train = X_train[:self.train_sample_size]
            y_train_min = y_train_min[:self.train_sample_size]
        if self.test_sample_size is not None:
            X_test = X_test[:self.test_sample_size]
            y_test_min = y_test_min[:self.test_sample_size]

        # Binary target for Stage 1
        y_train_bin = (y_train_min >= self.delay_threshold).astype(np.int32)
        y_test_bin  = (y_test_min  >= self.delay_threshold).astype(np.int32)

        # 3-class target for composite reporting
        def to_class(d):
            if d < 15: return 0
            elif d <= 30: return 1
            else: return 2
        y_train_3c = np.array([to_class(d) for d in y_train_min], dtype=np.int32)
        y_test_3c  = np.array([to_class(d) for d in y_test_min], dtype=np.int32)

        # Stage 2 training set: only true delayed flights
        train_delayed_mask = y_train_bin == 1
        X_train_delayed = X_train[train_delayed_mask]
        y_train_delayed = y_train_min[train_delayed_mask]

        self.X_train, self.X_test = X_train, X_test
        self.y_train_min, self.y_test_min = y_train_min, y_test_min
        self.y_train_bin, self.y_test_bin = y_train_bin, y_test_bin
        self.y_train_3c, self.y_test_3c   = y_train_3c, y_test_3c
        self.X_train_delayed = X_train_delayed
        self.y_train_delayed = y_train_delayed
        self.n_features = X_train.shape[1]

        n_on = int((y_train_bin == 0).sum())
        n_de = int((y_train_bin == 1).sum())
        print(f"Full train: {X_train.shape}  |  Test: {X_test.shape}")
        print(f"Stage 2 training subset (delayed only): {X_train_delayed.shape}")
        print(f"Train binary distribution: On-time={n_on} ({n_on/len(y_train_bin):.1%}),  "
              f"Delayed={n_de} ({n_de/len(y_train_bin):.1%})")

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------
    def _build_mlp(self, n_outputs, output_activation, hidden_units, dropout_rate,
                   learning_rate, loss, metrics, output_bias_init=None):
        """Shared MLP backbone used by both stages."""
        model = Sequential()
        model.add(Input(shape=(self.n_features,)))
        for i, units in enumerate(hidden_units):
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            rate = dropout_rate if i < len(hidden_units) - 1 else max(dropout_rate - 0.1, 0.0)
            if rate > 0:
                model.add(Dropout(rate))
        if output_bias_init is not None:
            model.add(Dense(n_outputs, activation=output_activation,
                            bias_initializer=Constant(output_bias_init)))
        else:
            model.add(Dense(n_outputs, activation=output_activation))
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss=loss, metrics=metrics)
        return model

    def build_stage1(self, hidden_units=(128, 64, 32), dropout_rate=0.3, learning_rate=1e-3):
        """Stage 1: binary classifier with a 2-output softmax head."""
        _, counts = np.unique(self.y_train_bin, return_counts=True)
        priors = counts / counts.sum()
        log_priors = np.log(priors).astype(np.float32)

        self.stage1_model = self._build_mlp(
            n_outputs=2,
            output_activation='softmax',
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            output_bias_init=log_priors,
        )
        print("\n--- Stage 1 Architecture (Binary Classifier) ---")
        self.stage1_model.summary()
        return self.stage1_model

    def build_stage2(self, hidden_units=(128, 64, 32), dropout_rate=0.3, learning_rate=1e-3):
        """Stage 2: regression on the delayed-only training subset."""
        self.stage2_model = self._build_mlp(
            n_outputs=1,
            output_activation=None,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            loss='mse',
            metrics=['mae'],
        )
        print("\n--- Stage 2 Architecture (Regression, delayed only) ---")
        self.stage2_model.summary()
        return self.stage2_model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_stage1(self, epochs=80, batch_size=512, validation_split=0.2,
                     patience=10, verbose=1):
        if self.stage1_model is None:
            self.build_stage1()

        classes = np.unique(self.y_train_bin)
        weights = compute_class_weight('balanced', classes=classes, y=self.y_train_bin)
        class_weight = dict(zip(classes, weights))
        print(f"Stage 1 class_weight = {class_weight}")

        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train_bin,
            test_size=validation_split,
            stratify=self.y_train_bin,
            random_state=self.random_state,
        )

        callbacks = [
            MacroF1Monitor(X_val, y_val, patience=patience, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=max(patience // 2, 3), verbose=1),
        ]

        print(f"\nTraining Stage 1 for up to {epochs} epochs (batch_size={batch_size})...")
        self.stage1_history = self.stage1_model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=verbose,
        )
        return self.stage1_history

    def train_stage2(self, epochs=80, batch_size=512, validation_split=0.2,
                     patience=10, verbose=1):
        if self.stage2_model is None:
            self.build_stage2()

        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train_delayed, self.y_train_delayed,
            test_size=validation_split,
            random_state=self.random_state,
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=max(patience // 2, 3), verbose=1),
        ]

        print(f"\nTraining Stage 2 (regression, delayed only) for up to {epochs} epochs...")
        self.stage2_history = self.stage2_model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose,
        )
        return self.stage2_history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_stage1(self, threshold=0.5):
        """Binary metrics on the full test set, with a tunable threshold."""
        probs = self.stage1_model.predict(self.X_test, verbose=0)
        prob_delayed = probs[:, 1]
        preds = (prob_delayed >= threshold).astype(np.int32)

        acc = accuracy_score(self.y_test_bin, preds)
        f1_pos = f1_score(self.y_test_bin, preds, pos_label=1, zero_division=0)
        macro_f1 = f1_score(self.y_test_bin, preds, average='macro', zero_division=0)
        cm = confusion_matrix(self.y_test_bin, preds)
        report = classification_report(self.y_test_bin, preds,
                                       target_names=self.CLASS_NAMES_2,
                                       zero_division=0)

        self.results['stage1'] = {
            'accuracy': acc, 'f1_delayed': f1_pos, 'macro_f1': macro_f1,
            'cm': cm, 'report': report, 'probs': prob_delayed, 'preds': preds,
            'threshold': threshold,
        }

        print(f"\n--- Stage 1 (Binary: On-time vs Delayed) | threshold={threshold} ---")
        print(f"Accuracy: {acc*100:.2f}%  |  F1 (Delayed): {f1_pos:.3f}  |  Macro F1: {macro_f1:.3f}")
        print(report)

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.CLASS_NAMES_2, yticklabels=self.CLASS_NAMES_2)
        plt.title(f'Stage 1 Confusion Matrix (threshold={threshold})')
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.tight_layout(); plt.show()
        return self.results['stage1']

    def evaluate_stage2(self):
        """Regression metrics on the TRUE delayed test flights only."""
        delayed_mask = self.y_test_bin == 1
        X_delayed = self.X_test[delayed_mask]
        y_delayed = self.y_test_min[delayed_mask]

        preds = self.stage2_model.predict(X_delayed, verbose=0).flatten()
        mse  = mean_squared_error(y_delayed, preds)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_delayed, preds)
        r2   = r2_score(y_delayed, preds)

        self.results['stage2'] = {
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'preds': preds, 'y_true': y_delayed,
        }

        print(f"\n--- Stage 2 (Regression on delayed-only test set) ---")
        print(f"N delayed test flights: {len(y_delayed)}")
        print(f"MSE: {mse:.2f}  RMSE: {rmse:.2f} min  MAE: {mae:.2f} min  R^2: {r2:.4f}")

        n_plot = min(2000, len(preds))
        idx = np.random.choice(len(preds), n_plot, replace=False)
        plt.figure(figsize=(6, 6))
        plt.scatter(y_delayed[idx], preds[idx], alpha=0.3, s=10)
        lo, hi = float(y_delayed[idx].min()), float(y_delayed[idx].max())
        plt.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Ideal')
        plt.xlabel('Actual ARR_DELAY (min, delayed only)')
        plt.ylabel('Predicted ARR_DELAY (min)')
        plt.title('Stage 2 Regression: Predicted vs Actual (delayed only)')
        plt.legend(); plt.tight_layout(); plt.show()
        return self.results['stage2']

    def evaluate_composite(self, threshold=0.5, ontime_minute_value=0.0):
        """
        End-to-end cascade evaluation.

        Builds the composite prediction by chaining Stage 1 -> Stage 2 and
        reports the metrics directly comparable to the flat MLP and the
        Ensemble experiments.
        """
        self.evaluate_stage1(threshold=threshold)

        stage1_preds = self.results['stage1']['preds']
        predicted_delayed = stage1_preds == 1

        if predicted_delayed.any():
            X_to_regress = self.X_test[predicted_delayed]
            stage2_preds_on_predicted = self.stage2_model.predict(X_to_regress, verbose=0).flatten()
        else:
            stage2_preds_on_predicted = np.array([], dtype=np.float32)

        # Composite minute predictions
        composite_minutes = np.full(len(self.X_test), ontime_minute_value, dtype=np.float32)
        composite_minutes[predicted_delayed] = stage2_preds_on_predicted

        # Composite 3-class predictions
        composite_3c = np.zeros(len(self.X_test), dtype=np.int32)
        delayed_indices = np.where(predicted_delayed)[0]
        # By default delayed -> class 2 (Long); overwrite to class 1 (Short) where minutes < 30
        composite_3c[delayed_indices] = 2
        bin_short = stage2_preds_on_predicted < 30
        composite_3c[delayed_indices[bin_short]] = 1

        acc_3c = accuracy_score(self.y_test_3c, composite_3c)
        f1_per = f1_score(self.y_test_3c, composite_3c, average=None, zero_division=0)
        macro_f1_3c = f1_score(self.y_test_3c, composite_3c, average='macro', zero_division=0)
        cm_3c = confusion_matrix(self.y_test_3c, composite_3c)
        report_3c = classification_report(self.y_test_3c, composite_3c,
                                          target_names=self.CLASS_NAMES_3,
                                          zero_division=0)

        mae_e2e  = mean_absolute_error(self.y_test_min, composite_minutes)
        rmse_e2e = np.sqrt(mean_squared_error(self.y_test_min, composite_minutes))

        self.results['composite'] = {
            'accuracy_3c': acc_3c,
            'macro_f1_3c': macro_f1_3c,
            'f1_ontime': f1_per[0], 'f1_short': f1_per[1], 'f1_long': f1_per[2],
            'cm_3c': cm_3c,
            'report_3c': report_3c,
            'mae_end_to_end': mae_e2e,
            'rmse_end_to_end': rmse_e2e,
            'composite_minutes': composite_minutes,
            'composite_3c': composite_3c,
        }

        print("\n--- COMPOSITE end-to-end (cascade: Stage 1 -> Stage 2) ---")
        print(f"Accuracy (3-class): {acc_3c*100:.2f}%")
        print(f"F1 per class: On-time={f1_per[0]:.2f}  Short={f1_per[1]:.2f}  Long={f1_per[2]:.2f}")
        print(f"Macro F1 (3-class): {macro_f1_3c:.3f}")
        print(f"End-to-end MAE: {mae_e2e:.2f} min  |  RMSE: {rmse_e2e:.2f} min")
        print(report_3c)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_3c, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.CLASS_NAMES_3, yticklabels=self.CLASS_NAMES_3)
        plt.title('Composite 3-class Confusion Matrix (Hierarchical MLP)')
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.tight_layout(); plt.show()
        return self.results['composite']

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def plot_histories(self):
        """Plot both stages' training curves side-by-side."""
        if self.stage1_history is None and self.stage2_history is None:
            print("Train at least one stage before plotting.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        if self.stage1_history is not None:
            h = self.stage1_history.history
            axes[0].plot(h['loss'], label='Train Loss', marker='o', markersize=3)
            axes[0].plot(h['val_loss'], label='Val Loss', marker='o', markersize=3)
            if 'val_macro_f1' in h:
                ax2 = axes[0].twinx()
                ax2.plot(h['val_macro_f1'], 'g-', marker='s', markersize=3,
                         label='Val Macro F1')
                ax2.set_ylabel('Macro F1', color='g')
                ax2.tick_params(axis='y', labelcolor='g')
            axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
            axes[0].set_title('Stage 1 - Binary Classifier')
            axes[0].legend(loc='upper right')
            axes[0].grid(alpha=0.3)

        if self.stage2_history is not None:
            h = self.stage2_history.history
            axes[1].plot(h['loss'], label='Train MSE', marker='o', markersize=3)
            axes[1].plot(h['val_loss'], label='Val MSE', marker='o', markersize=3)
            axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE')
            axes[1].set_title('Stage 2 - Regression (delayed only)')
            axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------
    # One-call pipeline
    # ------------------------------------------------------------------
    def run_all(self, hidden_units=(128, 64, 32), dropout_rate=0.3,
                learning_rate=1e-3, epochs=80, batch_size=512, patience=10,
                threshold=0.5, verbose=1):
        """Convenience: build, train, and evaluate both stages, then composite."""
        # Stage 1
        print("=" * 60)
        print("STAGE 1 - Binary Classifier (On-time vs Delayed)")
        print("=" * 60)
        self.build_stage1(hidden_units, dropout_rate, learning_rate)
        self.train_stage1(epochs=epochs, batch_size=batch_size,
                          patience=patience, verbose=verbose)

        # Stage 2
        print("\n" + "=" * 60)
        print("STAGE 2 - Regression on Delayed Flights")
        print("=" * 60)
        self.build_stage2(hidden_units, dropout_rate, learning_rate)
        self.train_stage2(epochs=epochs, batch_size=batch_size,
                          patience=patience, verbose=verbose)

        # Training curves + per-stage evaluation
        self.plot_histories()
        self.evaluate_stage2()

        # End-to-end composite (also runs evaluate_stage1 internally)
        print("\n" + "=" * 60)
        print("COMPOSITE - End-to-end cascade evaluation")
        print("=" * 60)
        self.evaluate_composite(threshold=threshold)
        return self.results
