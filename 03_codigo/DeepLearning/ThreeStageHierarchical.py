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

# Reuse the MacroF1Monitor: it operates on any softmax head with >= 2 outputs,
# so it works unchanged for the 3-class classifier in Stage 1.
from DeepLearning.DeepLearning import MacroF1Monitor


class ThreeStageHierarchical:
    """
    Three-stage hierarchical model proposed by the professor.

    Motivation:
        ~73% of flights are on-time, which dominates a single global regressor
        trained on ARR_DELAY (the flat regression head produced R^2 = -1.20).
        The two-stage cascade in HierarchicalDeepLearning.py already addresses
        this by splitting on-time vs delayed before regressing. The professor's
        suggestion goes one step further: split the *delayed* class into Short
        (15-30 min) and Long (>30 min), and train one dedicated regressor for
        each. The intuition is that the two delay regimes have very different
        target distributions (narrow, bounded vs wide, heavy-tailed) and a
        regressor specialised for each regime should be markedly more accurate
        than a single regressor covering [15, inf).

    Pipeline:
        Stage 1 - 3-class classifier (softmax over On-time / Short / Long).
                  Trained on the full training set with class_weight='balanced'
                  and the MacroF1Monitor callback (avoids majority collapse).
        Stage 2a - Regressor on the Short subset (15 <= ARR_DELAY <= 30).
                   Linear output, MSE loss.
        Stage 2b - Regressor on the Long subset (ARR_DELAY > 30).
                   Linear output, MSE loss.

    Inference:
        For each test flight:
            cls = argmax(stage1.predict(x))
            if cls == 0:   minutes = ontime_minute_value (default 0)
            elif cls == 1: minutes = stage2_short.predict(x)
            else:          minutes = stage2_long.predict(x)

    Composite evaluation (directly comparable to flat MLP and Hierarchical 2-stage):
        - 3-class metrics: Accuracy, per-class F1, Macro F1, confusion matrix
          (taken straight from Stage 1, since this model classifies natively).
        - End-to-end regression: MAE / RMSE on the full test set's ARR_DELAY.
        - Per-stage regression on the TRUE Short / Long test subsets to assess
          the specialisation gain isolated from classifier error.

    Why this is added on top of HierarchicalDeepLearning (binary + regress):
        - Provides a third hierarchical variant for the model comparison table.
        - The 3-class classifier in Stage 1 is the "natural" classification
          model: it does not require post-hoc binning of regressor outputs to
          recover Short vs Long, and it is comparable head-to-head with the
          flat 3-class MLP and the ensemble classifiers.
        - Each Stage-2 regressor sees a tight, well-defined target range,
          which (a) makes MSE easier to minimise and (b) lowers the variance
          of per-class predictions vs the single-regressor cascade.

    Attributes:
        data_loader, train_sample_size, test_sample_size, random_state -
            same conventions as DeepLearning / HierarchicalDeepLearning.
        short_lo / short_hi - thresholds defining the Short class (default 15-30).
        stage1_model, stage2_short_model, stage2_long_model - trained Keras models.
        stage1_history, stage2_short_history, stage2_long_history - histories.
        results - dict with keys 'stage1', 'stage2_short', 'stage2_long', 'composite'.
    """

    CLASS_NAMES_3 = ['On-time', 'Short', 'Long']

    def __init__(self, data_loader, train_sample_size=None, test_sample_size=None,
                 short_lo=15, short_hi=30, random_state=42):
        self.data_loader = data_loader
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.short_lo = short_lo
        self.short_hi = short_hi
        self.random_state = random_state

        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.stage1_model = None
        self.stage2_short_model = None
        self.stage2_long_model = None
        self.stage1_history = None
        self.stage2_short_history = None
        self.stage2_long_history = None
        self.results = {}

        self._prepare_data()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _to_class(self, d):
        if d < self.short_lo:
            return 0
        elif d <= self.short_hi:
            return 1
        else:
            return 2

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

        y_train_3c = np.array([self._to_class(d) for d in y_train_min], dtype=np.int32)
        y_test_3c  = np.array([self._to_class(d) for d in y_test_min], dtype=np.int32)

        # Stage 2a training subset: only Short-delay flights
        short_mask = y_train_3c == 1
        # Stage 2b training subset: only Long-delay flights
        long_mask = y_train_3c == 2

        self.X_train, self.X_test = X_train, X_test
        self.y_train_min, self.y_test_min = y_train_min, y_test_min
        self.y_train_3c, self.y_test_3c = y_train_3c, y_test_3c
        self.X_train_short = X_train[short_mask]
        self.y_train_short = y_train_min[short_mask]
        self.X_train_long  = X_train[long_mask]
        self.y_train_long  = y_train_min[long_mask]
        self.n_features = X_train.shape[1]

        n0 = int((y_train_3c == 0).sum())
        n1 = int((y_train_3c == 1).sum())
        n2 = int((y_train_3c == 2).sum())
        print(f"Full train: {X_train.shape}  |  Test: {X_test.shape}")
        print(f"Stage 2a training subset (Short, {self.short_lo}-{self.short_hi} min): "
              f"{self.X_train_short.shape}")
        print(f"Stage 2b training subset (Long,  >{self.short_hi} min): "
              f"{self.X_train_long.shape}")
        print(f"Train 3-class distribution: On-time={n0} ({n0/len(y_train_3c):.1%}),  "
              f"Short={n1} ({n1/len(y_train_3c):.1%}),  "
              f"Long={n2} ({n2/len(y_train_3c):.1%})")

    # ------------------------------------------------------------------
    # Shared MLP backbone
    # ------------------------------------------------------------------
    def _build_mlp(self, n_outputs, output_activation, hidden_units, dropout_rate,
                   learning_rate, loss, metrics, output_bias_init=None):
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

    # ------------------------------------------------------------------
    # Stage 1: 3-class classifier
    # ------------------------------------------------------------------
    def build_stage1(self, hidden_units=(128, 64, 32), dropout_rate=0.3, learning_rate=1e-3):
        _, counts = np.unique(self.y_train_3c, return_counts=True)
        priors = counts / counts.sum()
        log_priors = np.log(priors).astype(np.float32)

        self.stage1_model = self._build_mlp(
            n_outputs=3,
            output_activation='softmax',
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            output_bias_init=log_priors,
        )
        print("\n--- Stage 1 Architecture (3-class Classifier) ---")
        self.stage1_model.summary()
        return self.stage1_model

    def train_stage1(self, epochs=80, batch_size=512, validation_split=0.2,
                     patience=10, verbose=1):
        if self.stage1_model is None:
            self.build_stage1()

        classes = np.unique(self.y_train_3c)
        weights = compute_class_weight('balanced', classes=classes, y=self.y_train_3c)
        class_weight = dict(zip(classes, weights))
        print(f"Stage 1 class_weight = {class_weight}")

        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train_3c,
            test_size=validation_split,
            stratify=self.y_train_3c,
            random_state=self.random_state,
        )

        callbacks = [
            MacroF1Monitor(X_val, y_val, patience=patience, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=max(patience // 2, 3), verbose=1),
        ]

        print(f"\nTraining Stage 1 (3-class) for up to {epochs} epochs (batch_size={batch_size})...")
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

    # ------------------------------------------------------------------
    # Stage 2a / 2b: specialised regressors
    # ------------------------------------------------------------------
    def _build_regressor(self, hidden_units, dropout_rate, learning_rate):
        return self._build_mlp(
            n_outputs=1,
            output_activation=None,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            loss='mse',
            metrics=['mae'],
        )

    def build_stage2_short(self, hidden_units=(128, 64, 32), dropout_rate=0.3, learning_rate=1e-3):
        self.stage2_short_model = self._build_regressor(hidden_units, dropout_rate, learning_rate)
        print("\n--- Stage 2a Architecture (Regression, Short delays only) ---")
        self.stage2_short_model.summary()
        return self.stage2_short_model

    def build_stage2_long(self, hidden_units=(128, 64, 32), dropout_rate=0.3, learning_rate=1e-3):
        self.stage2_long_model = self._build_regressor(hidden_units, dropout_rate, learning_rate)
        print("\n--- Stage 2b Architecture (Regression, Long delays only) ---")
        self.stage2_long_model.summary()
        return self.stage2_long_model

    def _train_regressor(self, model, X, y, name, epochs, batch_size,
                         validation_split, patience, verbose):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.random_state,
        )
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=max(patience // 2, 3), verbose=1),
        ]
        print(f"\nTraining {name} for up to {epochs} epochs (n_train={len(X_tr)}, n_val={len(X_val)})...")
        return model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose,
        )

    def train_stage2_short(self, epochs=80, batch_size=512, validation_split=0.2,
                           patience=10, verbose=1):
        if self.stage2_short_model is None:
            self.build_stage2_short()
        self.stage2_short_history = self._train_regressor(
            self.stage2_short_model, self.X_train_short, self.y_train_short,
            "Stage 2a (Short regressor)",
            epochs, batch_size, validation_split, patience, verbose,
        )
        return self.stage2_short_history

    def train_stage2_long(self, epochs=80, batch_size=512, validation_split=0.2,
                          patience=10, verbose=1):
        if self.stage2_long_model is None:
            self.build_stage2_long()
        self.stage2_long_history = self._train_regressor(
            self.stage2_long_model, self.X_train_long, self.y_train_long,
            "Stage 2b (Long regressor)",
            epochs, batch_size, validation_split, patience, verbose,
        )
        return self.stage2_long_history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_stage1(self):
        """3-class classification metrics on the full test set."""
        probs = self.stage1_model.predict(self.X_test, verbose=0)
        preds = probs.argmax(axis=1)

        acc = accuracy_score(self.y_test_3c, preds)
        f1_per = f1_score(self.y_test_3c, preds, average=None, zero_division=0)
        macro_f1 = f1_score(self.y_test_3c, preds, average='macro', zero_division=0)
        cm = confusion_matrix(self.y_test_3c, preds)
        report = classification_report(self.y_test_3c, preds,
                                       target_names=self.CLASS_NAMES_3,
                                       zero_division=0)

        self.results['stage1'] = {
            'accuracy': acc,
            'f1_ontime': f1_per[0], 'f1_short': f1_per[1], 'f1_long': f1_per[2],
            'macro_f1': macro_f1, 'cm': cm, 'report': report,
            'probs': probs, 'preds': preds,
        }

        print(f"\n--- Stage 1 (3-class Classifier on full test set) ---")
        print(f"Accuracy: {acc*100:.2f}%  |  Macro F1: {macro_f1:.3f}")
        print(f"F1 per class: On-time={f1_per[0]:.2f}  Short={f1_per[1]:.2f}  Long={f1_per[2]:.2f}")
        print(report)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.CLASS_NAMES_3, yticklabels=self.CLASS_NAMES_3)
        plt.title('Stage 1 Confusion Matrix (3-class)')
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.tight_layout(); plt.show()
        return self.results['stage1']

    def _evaluate_regressor(self, model, X, y, key, label):
        if len(X) == 0:
            print(f"No test flights for {label} — skipping.")
            self.results[key] = None
            return None

        preds = model.predict(X, verbose=0).flatten()
        mse  = mean_squared_error(y, preds)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y, preds)
        r2   = r2_score(y, preds) if len(y) > 1 and np.var(y) > 0 else float('nan')

        self.results[key] = {
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'preds': preds, 'y_true': y,
        }

        print(f"\n--- {label} (n={len(y)}) ---")
        print(f"MSE: {mse:.2f}  RMSE: {rmse:.2f} min  MAE: {mae:.2f} min  R^2: {r2:.4f}")
        return self.results[key]

    def evaluate_stage2_short(self):
        mask = self.y_test_3c == 1
        return self._evaluate_regressor(
            self.stage2_short_model, self.X_test[mask], self.y_test_min[mask],
            'stage2_short', 'Stage 2a (Short regressor, TRUE Short test subset)',
        )

    def evaluate_stage2_long(self):
        mask = self.y_test_3c == 2
        return self._evaluate_regressor(
            self.stage2_long_model, self.X_test[mask], self.y_test_min[mask],
            'stage2_long', 'Stage 2b (Long regressor, TRUE Long test subset)',
        )

    def evaluate_composite(self, ontime_minute_value=0.0):
        """
        End-to-end cascade: route each test flight through Stage 1, then
        through Stage 2a / 2b according to the predicted class, and report
        end-to-end MAE / RMSE on the full test set.
        """
        if 'stage1' not in self.results:
            self.evaluate_stage1()
        cls_preds = self.results['stage1']['preds']

        composite_minutes = np.full(len(self.X_test), ontime_minute_value, dtype=np.float32)

        short_idx = np.where(cls_preds == 1)[0]
        long_idx  = np.where(cls_preds == 2)[0]

        if len(short_idx) > 0:
            composite_minutes[short_idx] = self.stage2_short_model.predict(
                self.X_test[short_idx], verbose=0).flatten()
        if len(long_idx) > 0:
            composite_minutes[long_idx] = self.stage2_long_model.predict(
                self.X_test[long_idx], verbose=0).flatten()

        mae  = mean_absolute_error(self.y_test_min, composite_minutes)
        rmse = np.sqrt(mean_squared_error(self.y_test_min, composite_minutes))

        self.results['composite'] = {
            'mae_end_to_end': mae,
            'rmse_end_to_end': rmse,
            'composite_minutes': composite_minutes,
            'n_routed_short': len(short_idx),
            'n_routed_long': len(long_idx),
            'n_routed_ontime': int((cls_preds == 0).sum()),
        }

        print("\n--- COMPOSITE end-to-end (Stage 1 -> Stage 2a / 2b) ---")
        print(f"Routed to On-time : {self.results['composite']['n_routed_ontime']}")
        print(f"Routed to Stage 2a: {len(short_idx)}")
        print(f"Routed to Stage 2b: {len(long_idx)}")
        print(f"End-to-end MAE : {mae:.2f} min")
        print(f"End-to-end RMSE: {rmse:.2f} min")
        return self.results['composite']

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_histories(self):
        if (self.stage1_history is None and
                self.stage2_short_history is None and
                self.stage2_long_history is None):
            print("Train at least one stage before plotting.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

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
            axes[0].set_title('Stage 1 - 3-class Classifier')
            axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
            axes[0].legend(loc='upper right'); axes[0].grid(alpha=0.3)

        for ax, hist, title in [
            (axes[1], self.stage2_short_history, 'Stage 2a - Short Regressor'),
            (axes[2], self.stage2_long_history,  'Stage 2b - Long Regressor'),
        ]:
            if hist is None:
                ax.set_title(f'{title} (not trained)')
                continue
            h = hist.history
            ax.plot(h['loss'], label='Train MSE', marker='o', markersize=3)
            ax.plot(h['val_loss'], label='Val MSE', marker='o', markersize=3)
            ax.set_title(title)
            ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
            ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------
    # One-call pipeline
    # ------------------------------------------------------------------
    def run_all(self, hidden_units=(128, 64, 32), dropout_rate=0.3,
                learning_rate=1e-3, epochs=80, batch_size=512, patience=10,
                verbose=1):
        print("=" * 60)
        print("STAGE 1 - 3-class Classifier (On-time / Short / Long)")
        print("=" * 60)
        self.build_stage1(hidden_units, dropout_rate, learning_rate)
        self.train_stage1(epochs=epochs, batch_size=batch_size,
                          patience=patience, verbose=verbose)

        print("\n" + "=" * 60)
        print("STAGE 2a - Regression on Short delays (15-30 min)")
        print("=" * 60)
        self.build_stage2_short(hidden_units, dropout_rate, learning_rate)
        self.train_stage2_short(epochs=epochs, batch_size=batch_size,
                                patience=patience, verbose=verbose)

        print("\n" + "=" * 60)
        print("STAGE 2b - Regression on Long delays (>30 min)")
        print("=" * 60)
        self.build_stage2_long(hidden_units, dropout_rate, learning_rate)
        self.train_stage2_long(epochs=epochs, batch_size=batch_size,
                               patience=patience, verbose=verbose)

        self.plot_histories()

        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        self.evaluate_stage1()
        self.evaluate_stage2_short()
        self.evaluate_stage2_long()
        self.evaluate_composite()
        return self.results
