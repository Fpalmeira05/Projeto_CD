import pandas as pd



from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import math
import seaborn as sns
import matplotlib.pyplot as plt


class DataVisualization:
    """
    A class responsible for data visualization.

    Attributes:
        data_loader (DataLoader): An object of the DataLoader class containing the dataset.

    Methods:
        perform_visualization(): Performs data visualization.
    """
    def __init__(self, data_loader, features_to_use):
        """
        Initializes the DataVisualization class with a DataLoader object.
        """
        self.data_loader = data_loader
        self.features_to_use = features_to_use


    def perform_visualization(self):
        """
        Performs data visualization.
        """

        print("Data Visualization Plots:")

        # Boxplot
        self.plot_boxplot()

        # Ridgeplot
        self.plot_ridgeplot()

    def plot_boxplot(self, sample_size=100000):
        """
        Plots boxplot for all selected features using a safe subsample.
        """
        print(f"Sampling {sample_size} rows for Boxplots to save time...")
        actual_sample_size = min(sample_size, len(self.data_loader.data_train))

        # 1. Isolate only the features we actually care about
        data_sample = self.data_loader.data_train[self.features_to_use].sample(n=actual_sample_size, random_state=42)

        # Create a single figure and axis for all boxplots
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot boxplots for each feature
        sns.boxplot(data=data_sample, ax=ax)
        ax.set_title("Boxplot of all Selected Features")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Scaled Value")

        # Rotate labels so they don't overlap
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

    def plot_ridgeplot(self, sample_size=10000):
        """
        Plots overlapping densities (ridge plot) for all features.
        KDE is mathematically heavy, so we use a smaller default sample size.
        """
        print(f"Sampling {sample_size} rows for Ridgeplots to prevent KDE memory crashes...")
        actual_sample_size = min(sample_size, len(self.data_loader.data_train))
        data_sample = self.data_loader.data_train[self.features_to_use].sample(n=actual_sample_size, random_state=42)

        features = data_sample.columns
        num_plots = len(features)

        # Make the figure height dynamic based on how many features you have
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 1.5 * num_plots), sharex=True)

        # Safety catch in case you only pass 1 feature
        if num_plots == 1:
            axes = [axes]

        # Generate a gradient of darker colors for the plots
        cmap = plt.get_cmap('Blues')
        colors = [cmap(1 - i / (num_plots + 1)) for i in range(1, num_plots + 1)]

        # Plot overlapping densities for each numerical feature
        for i, (feature, color) in enumerate(zip(features, colors)):
            # Draw the smooth KDE curve
            sns.kdeplot(data=data_sample[feature], ax=axes[i], color=color, fill=True, linewidth=2)

            # Label formatting
            axes[i].set_ylabel(feature, rotation=0, labelpad=50, ha='right', va='center')
            axes[i].yaxis.set_label_coords(-0.05, 0.2)

            # Remove box structure and y-ticks for that clean "floating" ridgeplot look
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].set_yticks([])

            # Adjust plot aesthetics
        axes[-1].set_xlabel("Scaled Value")

        # Overlap the plots slightly to create the "Ridge" effect
        plt.subplots_adjust(hspace=-0.4)

        plt.show()