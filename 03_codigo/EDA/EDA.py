import pandas as pd



from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import math
import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    """
    A class responsible for exploratory data analysis.
    Attributes:
        data_loader (DataLoader): Object containing the dataset.
        num_cols : The numerical columns in the dataset.
        cat_cols : The categorical columns in the dataset.

    Methods:
        perform_eda(): Performs exploratory data analysis.
        plot_distributions(): Plots distributions of the features.
        plot_correlation_heatmap(): Plots a correlation heatmap between features and labels.
        plot_feature_importance(): Computes and visualizes feature importance using permutation importance.
    """

    def __init__(self, data_loader, num_cols, cat_cols):
        """
        Initializes the EDA class with a DataLoader object.
        """
        self.data_loader = data_loader
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.features_to_use = num_cols + cat_cols

    def perform_eda(self):
        """
        Performs exploratory data analysis.
        """
        print("Exploratory Data Analysis (EDA) Report:")
        print("--------------------------------------")

        # Summary statistics
        print("\nSummary Statistics for train data:")
        print(self.data_loader.data_train.describe())
        print("\nSummary Statistics for test data:")
        print(self.data_loader.data_test.describe())

        # Distribution analysis
        print("\nDistribution Analysis:")
        self._plot_distributions()

        # Correlation analysis
        print("\nCorrelation Analysis:")
        self._plot_correlation_heatmap()

        # Feature Importance analysis
        print("\nFeature Importance Analysis:")
        self._plot_feature_importance()

    def _plot_distributions(self, sample_size = 100000, cols = 3):
        """
        Plots distributions of the features.
        """
        print(f"Sampling {sample_size} rows for distribution plots to save time...")
        actual_sample_size = min(sample_size, len(self.data_loader.data_train))

        #Sample the collumns that matters
        data_sample = self.data_loader.data_train[self.features_to_use].sample(n=actual_sample_size, random_state=42)

        num_cols = len(data_sample.columns)
        rows = math.ceil(num_cols / cols)
        fig, axes = plt.subplots(rows,cols,figsize=(5*cols,4*rows))

        if num_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, feature in enumerate(data_sample.columns):
            sns.histplot(data=data_sample, x=feature, ax=axes[i], bins=30)
            axes[i].set_title(f"Distribution of {feature}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Frequency")
        #Hide the empty grid
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def _plot_correlation_heatmap(self):
        """
        Plots a correlation heatmap between numerical features and labels.
        """
        # It's best to only correlate continuous numerical columns.
        # If you defined num_cols in your main script, you can pass them in,
        # or just grab them dynamically:
        numerical_train = self.data_loader.data_train[self.num_cols]

        data_with_labels = pd.concat([numerical_train, self.data_loader.labels_train], axis=1)
        corr_matrix = data_with_labels.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title("Correlation Heatmap (Numerical Features vs ARR_DELAY)")
        plt.tight_layout()
        plt.show()

    def _plot_feature_importance(self, n_estimators=10, n_repeats=2, sample_size=50000):
        """
        Computes and visualizes feature importance using permutation importance
        on a subsample of the data to save memory and time.
        """
        print(f"Sampling {sample_size} rows for feature importance to save time...")
        actual_sample_size = min(sample_size, len(self.data_loader.data_train))

        #Only preprocessed features
        X_sample = self.data_loader.data_train[self.features_to_use].sample(n=actual_sample_size, random_state=42)
        y_sample = self.data_loader.labels_train.loc[X_sample.index]

        rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf_model.fit(X_sample, y_sample)

        result = permutation_importance(rf_model, X_sample, y_sample, n_repeats=n_repeats, random_state=42, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), X_sample.columns[sorted_idx])
        plt.xlabel('Permutation Importance')
        plt.title('Feature Importance (Random Forest Regressor)')
        plt.tight_layout()
        plt.show()



