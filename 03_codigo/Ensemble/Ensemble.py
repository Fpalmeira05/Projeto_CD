import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


class Ensemble:
    """
    Applies two ensemble models to the multiclass flight delay problem:
        - Bagging  -> RandomForestClassifier
        - Boosting -> HistGradientBoostingClassifier

    Target classes (from ARR_DELAY in minutes):
        0 -> On-time     (< 15 min)
        1 -> Short delay (15-30 min)
        2 -> Long delay  (> 30 min)

    Attributes:
        data_loader        : DataLoader object holding preprocessed train/test data.
        train_sample_size  : Optional cap on training rows (speeds up experiments).
        test_sample_size   : Optional cap on testing rows.
        random_state       : Seed for reproducibility.
        bagging_model      : Trained RandomForestClassifier (after train_bagging()).
        boosting_model     : Trained HistGradientBoostingClassifier (after train_boosting()).
        results            : Dict with accuracy / classification report / confusion matrix per model.

    Methods:
        train_bagging  : Trains and evaluates the Random Forest model.
        train_boosting : Trains and evaluates the Hist Gradient Boosting model.
        run_all        : Trains and evaluates both ensemble models.
    """

    CLASS_NAMES = ['On-time', 'Short', 'Long']

    def __init__(self, data_loader, train_sample_size=None, test_sample_size=None,
                 balance_training=False, random_state=42):
        self.data_loader = data_loader
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.balance_training = balance_training
        self.random_state = random_state

        self.bagging_model = None
        self.boosting_model = None
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
        # Drop non-numeric columns (same approach used in the KNN section)
        cols_to_drop = self.data_loader.data_train.select_dtypes(exclude=['number', 'bool']).columns
        if len(cols_to_drop) > 0:
            print(f"Dropping non-numeric columns: {list(cols_to_drop)}")

        X_train = self.data_loader.data_train.drop(columns=cols_to_drop).values.astype(np.float32)
        X_test = self.data_loader.data_test.drop(columns=cols_to_drop).values.astype(np.float32)

        y_train = self.data_loader.labels_train.apply(self._categorize_delay).values
        y_test = self.data_loader.labels_test.apply(self._categorize_delay).values

        if self.train_sample_size is not None:
            X_train = X_train[:self.train_sample_size]
            y_train = y_train[:self.train_sample_size]
        if self.test_sample_size is not None:
            X_test = X_test[:self.test_sample_size]
            y_test = y_test[:self.test_sample_size]

        if self.balance_training:
            rng = np.random.default_rng(self.random_state)
            classes, counts = np.unique(y_train, return_counts=True)
            min_count = counts.min()
            balanced_idx = np.concatenate([
                rng.choice(np.where(y_train == c)[0], size=min_count, replace=False)
                for c in classes
            ])
            rng.shuffle(balanced_idx)
            X_train, y_train = X_train[balanced_idx], y_train[balanced_idx]
            print(f"Class distribution after balancing: { {self.CLASS_NAMES[c]: int(min_count) for c in classes} }")

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        print(f"Train set: {self.X_train.shape}  |  Test set: {self.X_test.shape}")

    def train_bagging(self, n_estimators=100, max_depth=None):
        """Random Forest = bagging of decision trees with feature subsampling."""
        print(f"\n[Bagging] Training RandomForestClassifier (n_estimators={n_estimators})...")
        self.bagging_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.bagging_model.fit(self.X_train, self.y_train)
        preds = self.bagging_model.predict(self.X_test)
        self._report("Random Forest (Bagging)", preds)

    def train_boosting(self, max_iter=200, learning_rate=0.1, max_depth=None):
        """HistGradientBoosting = sequential boosted trees, fast on large tabular data."""
        print(f"\n[Boosting] Training HistGradientBoostingClassifier (max_iter={max_iter})...")
        self.boosting_model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
        )
        self.boosting_model.fit(self.X_train, self.y_train)
        preds = self.boosting_model.predict(self.X_test)
        self._report("HistGradientBoosting (Boosting)", preds)

    def _report(self, name, preds):
        acc = accuracy_score(self.y_test, preds)
        report = classification_report(self.y_test, preds, target_names=self.CLASS_NAMES, zero_division=0)
        cm = confusion_matrix(self.y_test, preds)
        f1_per_class = f1_score(self.y_test, preds, average=None, zero_division=0)
        macro_f1 = f1_score(self.y_test, preds, average='macro', zero_division=0)

        self.results[name] = {
            'accuracy': acc,
            'report': report,
            'cm': cm,
            'f1_ontime': f1_per_class[0],
            'f1_short': f1_per_class[1],
            'f1_long': f1_per_class[2],
            'macro_f1': macro_f1,
        }

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc * 100:.2f}%")
        print(report)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.CLASS_NAMES, yticklabels=self.CLASS_NAMES)
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

    def run_all(self):
        self.train_bagging()
        self.train_boosting()