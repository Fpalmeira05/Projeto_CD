import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
import seaborn as sns
from sklearn.manifold import TSNE
import math


class FlightDimensionalityReduction:
    """
    Class responsible for the dimension reduction for the Flight Delay dataset.

    Attributes:
        data_loader (DataLoader): Object containing the dataset.
        features_to_use : features to use for dimensionality reduction.
        sample_size: Sample of the dataset.

    """

    def __init__(self, data_loader, features_to_use, sample_size=10000):
        """
        Initialize the object with a safe random sample of the dataset.
        """
        self.data_loader = data_loader

        print(f"Sampling {sample_size} rows for dimensionality reduction...")
        actual_sample_size = min(sample_size, len(self.data_loader.data_train))

        # Isolate the preprocessed features
        self.X_sample = self.data_loader.data_train[features_to_use].sample(n=actual_sample_size, random_state=42)
        self.y_sample = self.data_loader.labels_train.loc[self.X_sample.index]

        # Create a binary label for clear visual clustering (1 = Delayed > 15 mins, 0 = On Time)
        self.y_binary = (self.y_sample > 15).astype(int)

    def compute_pca(self, n_components=2):
        """
        Compute Principal Component Analysis (PCA) on the dataset.
        """
        print("Computing PCA (Linear Method)...")
        return PCA(n_components=n_components).fit_transform(self.X_sample)

    def compute_lda(self, n_components=2):
        """
        Perform Linear Discriminant Analysis (LDA) on the input data.

        Parameters:
        - n_components: The number of components to keep

        Returns:
            array-like: The reduced-dimensional representation of the data using LDA.
        """
        return LinearDiscriminantAnalysis(n_components=n_components).fit_transform(self.data, self.targets)


    def compute_umap(self, n_components=2, n_neighbors=15, min_dist=0.1):
        """
        Compute Uniform Manifold Approximation and Projection (UMAP).
        """
        print("Computing UMAP (Non-Linear Method)...")
        return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                         random_state=42).fit_transform(self.X_sample)

    def compute_tsne(self, n_components=2, perplexity=30):
        """
        Compute t-SNE (Included as a non-linear backup in case UMAP fails).
        """
        print("Computing t-SNE (Non-Linear Method)...")
        return TSNE(n_components=n_components, perplexity=perplexity, random_state=42).fit_transform(self.X_sample)

    def plot_projection(self, projection, title):
        """
        Plot the 2D projection of the dataset.
        """
        plt.figure(figsize=(10, 8))

        # We use seaborn instead of standard matplotlib to easily map our binary labels to colors
        scatter = sns.scatterplot(
            x=projection[:, 0],
            y=projection[:, 1],
            hue=self.y_binary,
            palette={0: 'blue', 1: 'red'},  # Blue = On time, Red = Delayed
            alpha=0.6,
            s=20
        )

        # Customize the legend to read "On Time" and "Delayed" instead of 0 and 1
        handles, _ = scatter.get_legend_handles_labels()
        plt.legend(handles=handles, title='Flight Status', labels=['On Time (<=15m)', 'Delayed (>15m)'])

        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def analyze_umap_clusters(self, umap_proj, features_to_check):
        """
        Re-plots the UMAP projection, coloring the dots by different features
        in a dynamic grid layout.
        """
        print(f"Generating UMAPs for {len(features_to_check)} features...")

        # Ensure we only try to plot features that actually exist in the data
        valid_features = [col for col in features_to_check if col in self.X_sample.columns]
        num_plots = len(valid_features)

        # Grid math: 3 plots per row
        cols = 3
        rows = math.ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))

        if num_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, feature in enumerate(valid_features):
            sns.scatterplot(
                x=umap_proj[:, 0],
                y=umap_proj[:, 1],
                hue=self.X_sample[feature],
                palette='viridis',
                alpha=0.6,
                s=15,
                ax=axes[i],
                legend=False
            )
            axes[i].set_title(f'UMAP Colored by:\n{feature}')
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        # Hide any empty grid spaces if your feature count isn't a multiple of 3
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()