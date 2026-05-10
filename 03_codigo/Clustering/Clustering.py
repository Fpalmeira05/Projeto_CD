import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.cluster.hierarchy import dendrogram, linkage


class Clustering:
    """
    Operational-profile clustering of airports or airlines.

    The dataset is flight-level, but the clustering task asks for patterns at
    the entity level: an *airport* (or *airline*) is described by the
    aggregate statistics of all the flights it operates. This class:

    1. Aggregates the flight-level data to one row per entity, producing a
       feature matrix with traffic-, delay-, distance-, route- and
       temporal-pattern features.
    2. Standard-scales the features.
    3. Applies three clustering algorithms (the project requires at least
       two; we use three to give the report a richer comparison):
           - K-Means with k = 2..8  (varying k as required by the spec).
           - DBSCAN with a sweep over `eps` (density-based, handles outliers).
           - Agglomerative (Ward) hierarchical, plus a dendrogram for the
             qualitative story.
    4. Reports per-algorithm cluster quality (silhouette, Davies-Bouldin,
       Calinski-Harabasz, inertia/elbow for K-Means) and visualises clusters
       on a 2-D PCA projection.
    5. Summarises each cluster's feature means and lists the top entities
       inside each cluster, so the cluster identities can be named
       interpretively in the report (e.g. 'small regional, high-punctuality').

    Expects a *cleaned* DataLoader (i.e. `data_loader_cleaned.pkl`) in which
    `ORIGIN` and `AIRLINE_CODE` are still raw string codes — not one-hot
    encoded.

    Attributes:
        data_loader   : The cleaned DataLoader.
        entity        : 'airport' or 'airline'.
        sample_size   : Optional cap on flight rows used for aggregation.
        min_flights   : Drop entities with fewer than this many flights
                        (avoids singleton clusters / unstable statistics).
        random_state  : Seed for reproducibility.
        features_df   : Aggregated feature DataFrame, indexed by entity code.
        X_scaled      : StandardScaled numpy feature matrix.
        results       : Dict of {algorithm -> details}.

    Methods:
        kmeans_sweep       : K-Means for a range of k, with eval plots.
        dbscan_sweep       : DBSCAN for a range of eps.
        agglomerative      : Single agglomerative run + dendrogram.
        visualize_clusters : 2-D PCA scatter, colored by cluster.
        summarize_clusters : Per-cluster feature means + member entities.
        run_all            : Full pipeline (all 3 algorithms).
    """

    def __init__(self, data_loader, entity='airport', sample_size=None,
                 min_flights=30, random_state=42):
        self.data_loader = data_loader
        self.entity = entity
        self.sample_size = sample_size
        self.min_flights = min_flights
        self.random_state = random_state
        self.results = {}

        self._prepare_features()
        self._scale_features()

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------
    def _prepare_features(self):
        if self.entity == 'airport':
            entity_col = 'ORIGIN'
        elif self.entity == 'airline':
            entity_col = 'AIRLINE_CODE'
        else:
            raise ValueError(
                f"Unknown entity '{self.entity}'. Expected 'airport' or 'airline'."
            )

        # Combine features + label into a single working DataFrame
        df = self.data_loader.data_train.copy()
        df['ARR_DELAY'] = self.data_loader.labels_train.values

        if self.sample_size is not None:
            df = df.iloc[:self.sample_size].copy()

        if entity_col not in df.columns:
            raise ValueError(
                f"Column '{entity_col}' not found in data_loader.data_train. "
                f"Make sure you are passing the CLEANED data_loader "
                f"(before one-hot encoding). Available columns sample: "
                f"{list(df.columns)[:10]}"
            )

        # ---- Core delay/traffic/distance aggregations ----
        agg = df.groupby(entity_col).agg({
            'ARR_DELAY':        ['count', 'mean', 'std', 'median'],
            'DISTANCE':         'mean',
            'CRS_ELAPSED_TIME': 'mean',
        })
        agg.columns = [
            'n_flights', 'mean_arr_delay', 'std_arr_delay', 'median_arr_delay',
            'mean_distance', 'mean_crs_elapsed',
        ]

        # ---- Delay-class proportions ----
        def categorize(d):
            if d < 15: return 'ontime'
            elif d <= 30: return 'short'
            else: return 'long'
        df['_class'] = df['ARR_DELAY'].apply(categorize)
        class_props = (
            df.groupby([entity_col, '_class']).size()
              .unstack(fill_value=0)
        )
        class_props = class_props.div(class_props.sum(axis=1), axis=0)
        for col in ['ontime', 'short', 'long']:
            if col not in class_props.columns:
                class_props[col] = 0.0
        class_props = class_props.rename(columns={
            'ontime': 'pct_ontime', 'short': 'pct_short', 'long': 'pct_long',
        })[['pct_ontime', 'pct_short', 'pct_long']]
        agg = agg.join(class_props)

        # ---- Temporal proportions (optional features) ----
        for col_in, col_out in [
            ('IS_WEEKEND', 'pct_weekend'),
            ('IS_HOLIDAY_MONTH', 'pct_holiday_month'),
        ]:
            if col_in in df.columns:
                temp = df.groupby(entity_col)[col_in].mean().rename(col_out)
                agg = agg.join(temp)

        # ---- Network / route diversity ----
        if self.entity == 'airport':
            n_dest = df.groupby(entity_col)['DEST'].nunique().rename('n_destinations')
            agg = agg.join(n_dest)
            if 'AIRLINE_CODE' in df.columns:
                n_air = df.groupby(entity_col)['AIRLINE_CODE'].nunique().rename('n_airlines')
                agg = agg.join(n_air)
        else:  # airline
            n_orig = df.groupby(entity_col)['ORIGIN'].nunique().rename('n_origins')
            n_dest = df.groupby(entity_col)['DEST'].nunique().rename('n_destinations')
            agg = agg.join(n_orig).join(n_dest)

        # ---- Filter low-traffic entities ----
        before = len(agg)
        agg = agg[agg['n_flights'] >= self.min_flights].copy()
        print(f"Aggregated {before} {self.entity}s; kept {len(agg)} with "
              f">= {self.min_flights} flights.")

        # NaN from std on singleton -> 0
        agg = agg.fillna(0.0)

        self.features_df = agg
        self.feature_names = list(agg.columns)
        self.entity_codes = list(agg.index)

        print(f"Feature matrix shape: {agg.shape}")
        print(f"Features used: {self.feature_names}")

    def _scale_features(self):
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.features_df.values)
        self.scaler = scaler

    # ------------------------------------------------------------------
    # K-Means
    # ------------------------------------------------------------------
    def kmeans_sweep(self, k_values=range(2, 9)):
        """K-Means for a range of k, plotting elbow + silhouette + DB index."""
        k_values = list(k_values)
        inertias, sils, dbs, chs = [], [], [], []
        all_labels = {}

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(self.X_scaled)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(self.X_scaled, labels))
            dbs.append(davies_bouldin_score(self.X_scaled, labels))
            chs.append(calinski_harabasz_score(self.X_scaled, labels))
            all_labels[k] = labels

        # Plot evaluation curves
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(k_values, inertias, 'bo-')
        axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow plot (inertia)')
        axes[0].grid(alpha=0.3)

        axes[1].plot(k_values, sils, 'go-')
        axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette')
        axes[1].set_title('Silhouette score  (higher is better)')
        axes[1].grid(alpha=0.3)

        axes[2].plot(k_values, dbs, 'ro-')
        axes[2].set_xlabel('k'); axes[2].set_ylabel('Davies-Bouldin')
        axes[2].set_title('Davies-Bouldin index  (lower is better)')
        axes[2].grid(alpha=0.3)

        plt.suptitle(f'K-Means sweep over k  ({self.entity})')
        plt.tight_layout(); plt.show()

        # Pick best k by silhouette
        best_idx = int(np.argmax(sils))
        best_k = k_values[best_idx]
        print(f"\nBest k by silhouette: k={best_k}  "
              f"(silhouette={sils[best_idx]:.3f}, "
              f"DB={dbs[best_idx]:.3f}, "
              f"CH={chs[best_idx]:.0f})")

        # Numeric summary table
        summary = pd.DataFrame({
            'k': k_values,
            'inertia': [f"{x:.1f}" for x in inertias],
            'silhouette': [f"{x:.3f}" for x in sils],
            'davies_bouldin': [f"{x:.3f}" for x in dbs],
            'calinski_harabasz': [f"{x:.0f}" for x in chs],
        }).set_index('k')
        print("\nK-Means sweep summary:")
        print(summary)

        self.results['kmeans'] = {
            'k_values': k_values,
            'inertias': inertias,
            'silhouettes': sils,
            'db_scores': dbs,
            'ch_scores': chs,
            'all_labels': all_labels,
            'best_k': best_k,
            'best_labels': all_labels[best_k],
            'summary': summary,
        }
        return self.results['kmeans']

    # ------------------------------------------------------------------
    # DBSCAN
    # ------------------------------------------------------------------
    def dbscan_sweep(self, eps_values=None, min_samples=5):
        """DBSCAN over a range of eps; report n_clusters, n_noise, silhouette."""
        if eps_values is None:
            eps_values = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

        n_clusters_list, n_noise_list, sils = [], [], []
        all_labels = {}

        for eps in eps_values:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(self.X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            n_clusters_list.append(n_clusters)
            n_noise_list.append(n_noise)

            # Silhouette only defined for >= 2 non-noise clusters
            mask = labels != -1
            if mask.sum() >= 2 and len(set(labels[mask])) >= 2:
                sils.append(silhouette_score(self.X_scaled[mask], labels[mask]))
            else:
                sils.append(np.nan)
            all_labels[eps] = labels

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(eps_values, n_clusters_list, 'bo-', label='# clusters')
        axes[0].plot(eps_values, n_noise_list, 'r^-', label='# noise points')
        axes[0].set_xlabel('eps'); axes[0].set_ylabel('Count')
        axes[0].set_title('DBSCAN: clusters and noise vs eps')
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(eps_values, sils, 'go-')
        axes[1].set_xlabel('eps'); axes[1].set_ylabel('Silhouette (non-noise)')
        axes[1].set_title('DBSCAN silhouette vs eps')
        axes[1].grid(alpha=0.3)

        plt.suptitle(f'DBSCAN sweep over eps  ({self.entity}, min_samples={min_samples})')
        plt.tight_layout(); plt.show()

        # Pick best eps by silhouette
        valid = [(i, s) for i, s in enumerate(sils) if not np.isnan(s)]
        if valid:
            best_i = max(valid, key=lambda t: t[1])[0]
            best_eps = eps_values[best_i]
            print(f"\nBest eps by silhouette: eps={best_eps}  "
                  f"(n_clusters={n_clusters_list[best_i]}, "
                  f"silhouette={sils[best_i]:.3f}, "
                  f"n_noise={n_noise_list[best_i]})")
        else:
            best_eps = eps_values[0]
            print("\nDBSCAN did not produce >=2 clusters at any eps. "
                  f"Defaulting to eps={best_eps}.")

        summary = pd.DataFrame({
            'eps': eps_values,
            'n_clusters': n_clusters_list,
            'n_noise': n_noise_list,
            'silhouette': [f"{x:.3f}" if not np.isnan(x) else 'n/a' for x in sils],
        }).set_index('eps')
        print("\nDBSCAN sweep summary:")
        print(summary)

        self.results['dbscan'] = {
            'eps_values': list(eps_values),
            'n_clusters_per_eps': n_clusters_list,
            'n_noise_per_eps': n_noise_list,
            'silhouettes': sils,
            'all_labels': all_labels,
            'best_eps': best_eps,
            'best_labels': all_labels[best_eps],
            'summary': summary,
        }
        return self.results['dbscan']

    # ------------------------------------------------------------------
    # Agglomerative
    # ------------------------------------------------------------------
    def agglomerative(self, n_clusters=4):
        """Run Agglomerative (Ward) + plot dendrogram (truncated)."""
        ac = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = ac.fit_predict(self.X_scaled)
        sil = silhouette_score(self.X_scaled, labels)
        db = davies_bouldin_score(self.X_scaled, labels)

        Z = linkage(self.X_scaled, method='ward')
        plt.figure(figsize=(14, 5))
        dendrogram(Z, truncate_mode='lastp', p=30, show_leaf_counts=True,
                   leaf_rotation=45, leaf_font_size=8)
        plt.title(f'Agglomerative (Ward) Dendrogram  ({self.entity}, last 30 merges)')
        plt.xlabel('Cluster size / entity index')
        plt.ylabel('Ward distance')
        plt.tight_layout(); plt.show()

        print(f"Agglomerative (k={n_clusters}):  silhouette = {sil:.3f}  |  "
              f"davies_bouldin = {db:.3f}")

        self.results['agglomerative'] = {
            'n_clusters': n_clusters,
            'labels': labels,
            'silhouette': sil,
            'davies_bouldin': db,
            'linkage_matrix': Z,
        }
        return self.results['agglomerative']

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def visualize_clusters(self, labels, title='Clusters'):
        """2-D PCA projection colored by cluster (noise points greyed out)."""
        pca = PCA(n_components=2, random_state=self.random_state)
        proj = pca.fit_transform(self.X_scaled)

        plt.figure(figsize=(9, 6))
        unique_labels = sorted(set(labels))
        palette = sns.color_palette('tab10', n_colors=max(10, len(unique_labels)))
        for lbl in unique_labels:
            mask = labels == lbl
            if lbl == -1:
                color, label_str = (0.6, 0.6, 0.6), 'Noise'
            else:
                color, label_str = palette[lbl % len(palette)], f'Cluster {lbl}'
            plt.scatter(proj[mask, 0], proj[mask, 1], s=45, alpha=0.75,
                        label=label_str, color=color,
                        edgecolor='k', linewidth=0.4)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.title(title)
        plt.legend(loc='best', fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------
    # Per-cluster summary
    # ------------------------------------------------------------------
    def summarize_clusters(self, labels, top_n_members=5):
        """Per-cluster feature means + representative members, ignoring noise."""
        df = self.features_df.copy()
        df['cluster'] = labels

        non_noise = df[df['cluster'] != -1]
        if non_noise.empty:
            print("All points are noise; no clusters to summarise.")
            return None

        means = non_noise.groupby('cluster').mean().round(2)
        print("\n--- Cluster feature means ---")
        print(means.T)

        sizes = df.groupby('cluster').size().rename('n_entities')
        print(f"\n--- Cluster sizes ---\n{sizes}")

        print(f"\n--- Top {top_n_members} {self.entity}s per cluster "
              f"(by n_flights) ---")
        for c in sorted(df['cluster'].unique()):
            if c == -1:
                continue
            members = df[df['cluster'] == c].sort_values('n_flights', ascending=False)
            top = list(members.head(top_n_members).index)
            print(f"  Cluster {c}: {', '.join(map(str, top))}")

        return means

    # ------------------------------------------------------------------
    # One-call pipeline
    # ------------------------------------------------------------------
    def run_all(self, k_values=range(2, 9), eps_values=None, agglo_k=4):
        """Run all three algorithms with evaluation, visualisation, summary."""
        print("=" * 60)
        print(f"1) K-MEANS  ({self.entity})")
        print("=" * 60)
        self.kmeans_sweep(k_values=k_values)
        best_k = self.results['kmeans']['best_k']
        self.visualize_clusters(
            self.results['kmeans']['best_labels'],
            title=f'K-Means clusters (k={best_k}, {self.entity})',
        )
        self.summarize_clusters(self.results['kmeans']['best_labels'])

        print("\n" + "=" * 60)
        print(f"2) DBSCAN  ({self.entity})")
        print("=" * 60)
        self.dbscan_sweep(eps_values=eps_values)
        best_eps = self.results['dbscan']['best_eps']
        self.visualize_clusters(
            self.results['dbscan']['best_labels'],
            title=f'DBSCAN clusters (eps={best_eps}, {self.entity})',
        )

        print("\n" + "=" * 60)
        print(f"3) AGGLOMERATIVE  ({self.entity})")
        print("=" * 60)
        self.agglomerative(n_clusters=agglo_k)
        self.visualize_clusters(
            self.results['agglomerative']['labels'],
            title=f'Agglomerative clusters (k={agglo_k}, {self.entity})',
        )
        self.summarize_clusters(self.results['agglomerative']['labels'])

        # Final comparison table
        print("\n" + "=" * 60)
        print(f"CLUSTERING COMPARISON  ({self.entity})")
        print("=" * 60)
        rows = [
            {
                'Algorithm': f"K-Means (k={best_k})",
                'n_clusters': best_k,
                'Silhouette': f"{max(self.results['kmeans']['silhouettes']):.3f}",
                'Davies-Bouldin': f"{self.results['kmeans']['db_scores'][self.results['kmeans']['k_values'].index(best_k)]:.3f}",
            },
            {
                'Algorithm': f"DBSCAN (eps={best_eps})",
                'n_clusters': self.results['dbscan']['n_clusters_per_eps'][
                    self.results['dbscan']['eps_values'].index(best_eps)
                ],
                'Silhouette': f"{self.results['dbscan']['silhouettes'][self.results['dbscan']['eps_values'].index(best_eps)]:.3f}"
                              if not np.isnan(self.results['dbscan']['silhouettes'][self.results['dbscan']['eps_values'].index(best_eps)])
                              else 'n/a',
                'Davies-Bouldin': 'n/a',
            },
            {
                'Algorithm': f"Agglomerative (k={agglo_k})",
                'n_clusters': agglo_k,
                'Silhouette': f"{self.results['agglomerative']['silhouette']:.3f}",
                'Davies-Bouldin': f"{self.results['agglomerative']['davies_bouldin']:.3f}",
            },
        ]
        comparison = pd.DataFrame(rows).set_index('Algorithm')
        print(comparison)
        self.results['comparison'] = comparison
        return self.results
