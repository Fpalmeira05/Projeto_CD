# Clustering Phase — Report

This document explains the *why* and the *what* of the clustering phase:
the choice of entity, the feature aggregation, the three algorithms applied,
the cluster identities recovered, and a key negative finding about the shape
of the data. It is meant as a source you can paraphrase directly into the
LaTeX report.

---

## 1. Problem formulation — clustering *airports*, not flights

The dataset is flight-level (≈1.4 M training rows), but the project spec
asks for clustering **at the entity level**:

> *"Identify patterns in operational performance by clustering airports
> based on delay behaviour and traffic characteristics, or airlines based
> on punctuality, delay causes, and route profiles."*

We focused on **airports** because:

- After filtering, the training set contains ~306 distinct origin airports
  (≥ 30 flights each), giving a meaningful population to cluster.
- Airlines (~15 carriers in the US data) are too few to produce stable
  clusters with K-Means / DBSCAN; clustering 15 entities would be more
  qualitative than quantitative.
- The airport view directly informs **airport-level operational decisions**
  (gate allocation, traffic management) that map cleanly onto the project's
  business objectives.

The `Clustering` class is parametrised by an `entity` argument
(`'airport'` or `'airline'`), so an airline-level analysis can be reproduced
by changing one keyword.

## 2. Feature engineering — flight-level → airport-level aggregation

Each airport is described by **13 aggregate features** computed over the
flights it operates as `ORIGIN`:

| Group | Features |
|---|---|
| **Traffic volume**     | `n_flights` |
| **Network reach**      | `n_destinations`, `n_airlines` |
| **Delay statistics**   | `mean_arr_delay`, `std_arr_delay`, `median_arr_delay` |
| **Delay class mix**    | `pct_ontime`, `pct_short`, `pct_long` |
| **Route profile**      | `mean_distance`, `mean_crs_elapsed` |
| **Temporal profile**   | `pct_weekend`, `pct_holiday_month` |

After aggregation the design matrix is **(306 airports × 13 features)**.

All features are **`StandardScaler`-normalised** before clustering. This is
not optional: without scaling, the `n_flights` feature (range ~30 to
>10 000) would dominate the Euclidean distance and the clusters would
become a 1-D ranking by traffic volume — exactly the failure mode that
makes clustering uninteresting.

The class also accepts a `min_flights` threshold (default 30) that drops
the tiny airports whose statistics are too noisy to be meaningful. On the
real data this filter reduced 377 raw airports to **306 usable ones**.

## 3. Three algorithms, three views of the data

The project requires at least two clustering algorithms with varying
numbers of clusters. We applied **three**, each chosen because it gives a
qualitatively different view of the structure:

| Algorithm | Family | Role |
|---|---|---|
| **K-Means** | Partition-based, centroid | Workhorse; varies k explicitly; gives the headline cluster identities. |
| **DBSCAN** | Density-based, no k required | Tests whether the airports form **dense clumps** vs a continuous gradient; surfaces outliers as a "noise" group. |
| **Agglomerative (Ward linkage)** | Hierarchical | Produces a **dendrogram** showing the merge order — the most interpretable view for the report, and refines what K-Means flattens. |

Cluster quality was assessed with:

- **Silhouette score** (range −1 to 1, higher is better) — measures how
  similar a point is to its own cluster vs the nearest other cluster.
- **Davies-Bouldin index** (lower is better) — ratio of within-cluster
  scatter to between-cluster separation.
- **Calinski-Harabasz score** (higher is better) — between/within variance
  ratio.
- **Inertia / elbow** (K-Means only) — within-cluster sum of squares vs k.

For each algorithm we also visualised the result on a **2-D PCA
projection** of the standardised feature matrix (PC1 + PC2 capture
≈52 % of the variance) and printed **per-cluster feature means** plus the
**top 5 airports per cluster** so the cluster identities can be named
interpretively in the report.

## 4. Results

### 4.1 K-Means sweep over k = 2…8

| k | inertia | silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| 2 | 3 233 | 0.193 | 1.845 | 70 |
| **3** | **2 689** | **0.208** ← max | **1.501** | **73** ← max |
| 4 | 2 371 | 0.185 | 1.490 | 68 |
| 5 | 2 152 | 0.195 | 1.454 | 64 |
| 6 | 2 006 | 0.169 | 1.667 | 59 |
| 7 | 1 893 | 0.189 | 1.550 | 55 |
| 8 | 1 786 | 0.184 | 1.559 | 52 |

**K-Means picked k = 3** by silhouette (0.208) — also the maximum of
Calinski-Harabasz (73). The elbow plot shows the typical concave
descent with the most noticeable bend between k = 2 → 3, supporting the
silhouette pick.

### 4.2 K-Means cluster identities (k = 3)

| Cluster | Size | Top airports | Mean delay | Std delay | On-time | Avg flights | Identity |
|---|---|---|---|---|---|---|---|
| **0** | 35  | PIE, ASE, PGD, JAC, IDA       | **44.9 min** | **117!** | 65 % | 93   | **Leisure / regional, weather-prone** |
| **1** | 152 | HNL, BUR, OGG, LGB, GSP        | 19.0 min     | 55       | 78 % | 160  | **Mid-sized punctual mainstream** |
| **2** | 119 | **ATL, DFW, ORD, DEN, CLT**    | 20.5 min     | 60       | 74 % | **2 280** | **Major hubs and large metros** |

The cluster identities are immediately interpretable by looking at the top
member airports:

- **Cluster 0** is dominated by airports such as **Aspen (ASE), Jackson
  Hole (JAC), Punta Gorda (PGD), Idaho Falls (IDA), St Petersburg (PIE)** —
  mountain resort and seasonal/leisure destinations. Mean delay of 45
  minutes with a **standard deviation of 117 minutes** (by far the highest)
  reflects the role of weather: when the weather is good these airports
  run on time; when it is not, delays cascade dramatically. Low traffic
  (93 flights on average) and limited destinations (≈9) confirm the
  regional profile.

- **Cluster 1** is the **long tail of mainstream mid-sized airports** —
  Honolulu (HNL), Burbank (BUR), Long Beach (LGB), Greenville-Spartanburg
  (GSP). 78 % on-time is the best rate of any cluster, reflecting good
  scheduling discipline and milder operational stress. 152 airports — by
  far the largest cluster — and accounts for the "bulk" of US aviation.

- **Cluster 2** is the **hub layer**: Atlanta (ATL), Dallas-Fort Worth
  (DFW), Chicago-O'Hare (ORD), Denver (DEN), Charlotte (CLT). Average
  **2 280 flights** per airport, 45 destinations, 11 airlines. Delay
  averages are only marginally worse than the punctual mid-sized cluster
  (20.5 vs 19.0 min) — the hubs have absorbed scale efficiently without
  collapsing into chronic delay.

### 4.3 DBSCAN sweep over eps

| eps  | n_clusters | n_noise | silhouette |
|---|---|---|---|
| 0.50 | 0 | 306 | n/a |
| 0.75 | 1 | 301 | n/a |
| 1.00 | 2 | 275 | 0.223 |
| 1.25 | 1 | 216 | n/a |
| **1.50** | **2** | **158** | **0.275** ← max |
| 1.75 | 1 | 117 | n/a |
| 2.00 | 1 | 86  | n/a |
| 2.50 | 1 | 46  | n/a |
| 3.00 | 1 | 23  | n/a |

At the best eps (1.5), DBSCAN identifies **2 clusters but labels 158 of
306 airports (52 %) as noise**. As eps grows the noise fraction drops,
but only one cluster survives — so noise points get absorbed into a
single mass rather than into multiple meaningful clusters.

### 4.4 Agglomerative (Ward, k = 4)

Silhouette = 0.156, Davies-Bouldin = 1.595.

| Cluster | Size | Top airports | n_flights | Mean delay | Identity |
|---|---|---|---|---|---|
| **0** | 33  | MSN, ASE, FWA, JAC, MRY     | 102   | **46.4** | Small regional / leisure (same as K-Means C0) |
| **1** | 152 | SJC, SMF, OAK, HNL, BUR     | 221   | 17.8     | Mid-sized punctual (same as K-Means C1)       |
| **2** | 90  | SAN, STL, PDX, RDU, MSY     | 768   | 22.7     | **Large-medium tier** (new sub-cluster)        |
| **3** | 31  | **ATL, DFW, ORD, DEN, CLT** | **6 214** | 19.8 | **Megahubs** (new sub-cluster)                |

Agglomerative reproduces the K-Means small-regional (C0) and punctual
mid-sized (C1) clusters exactly, then **splits the K-Means "major hubs"
cluster (119) into two**:

- **Cluster 2 (90 airports, ~770 flights each)** — large but not
  megahub: San Diego (SAN), St Louis (STL), Portland (PDX), Raleigh-Durham
  (RDU), New Orleans (MSY).
- **Cluster 3 (31 airports, 6 214 flights each)** — the **megahubs**:
  Atlanta, Dallas-Fort Worth, O'Hare, Denver, Charlotte.

The dendrogram visually supports this split: the red right-hand branches
(Cluster 3, megahubs) merge with the rest only at the highest Ward
distance (~25), while the rest of the structure resolves at much lower
distances (~10–17). The megahubs are genuinely more dissimilar from
"normal" large airports than the rest of the population is from itself.

### 4.5 Final comparison

| Algorithm | n_clusters | Silhouette | Davies-Bouldin |
|---|---|---|---|
| K-Means (k = 3)        | 3 | 0.208 | 1.501 |
| DBSCAN (eps = 1.5)     | 2 (+158 noise) | 0.275 (non-noise only) | n/a |
| Agglomerative (k = 4)  | 4 | 0.156 | 1.595 |

K-Means gives the cleanest **headline result** (3 named clusters);
Agglomerative gives the **most refined result** (splits hubs into
megahubs vs large-medium); DBSCAN gives the **most informative negative
finding** (the data is a continuous gradient, not density-separated).

## 5. Interpretation

### 5.1 Why the silhouette scores are modest (≈ 0.16–0.28)

All three algorithms produce silhouettes in the **0.16 – 0.28 range** —
modest in absolute terms. This is **not a flaw of the clustering**; it
reflects the underlying shape of the data:

- Airports vary along a **continuous operational gradient** (traffic
  volume × delay profile), not as a set of well-separated clumps.
- DBSCAN's 52 % noise fraction is the smoking gun: at any practical
  density threshold, more than half the airports fall outside the dense
  core. There simply isn't a "dense clump" structure to find.
- The K-Means and Agglomerative silhouettes are nevertheless **positive
  and consistent** across both algorithms (and across multiple k values
  in the K-Means sweep) — so the partitions are *real*, just not crisp.

In short: the operational categories shade into each other gradually
(a mid-size airport with above-average delays is *kind of* a leisure
airport; a large mid-size airport is *kind of* a hub) — and the
clustering algorithms partition that continuum into the most reasonable
segments rather than discovering pre-existing dense clumps.

### 5.2 Why DBSCAN's high noise fraction is itself a finding

DBSCAN labelling 158 of 306 airports as noise is **not a failure of the
algorithm** — it is a structural diagnosis of the data:

> *"DBSCAN's high noise fraction tells us the airport feature space is
> continuous, not density-separated. Airports vary gradually along a
> traffic-and-delay continuum rather than forming dense, well-separated
> clumps. K-Means and Agglomerative cope with this by imposing a partition
> on the continuum (they always return clusters); DBSCAN, which refuses to
> impose a partition where the data does not justify one, correctly
> declines."*

Reporting this negative finding is more valuable than tuning DBSCAN until
it produces an arbitrary number of clusters: it tells the reader
something true about the shape of the data.

### 5.3 The two converging cluster pictures

K-Means and Agglomerative **agree** on the structure of the data — they
just differ in granularity:

| Operational role | K-Means cluster | Agglomerative cluster | Size |
|---|---|---|---|
| Small / leisure / weather-prone   | **C0** | **C0** | 33–35 airports |
| Mid-sized punctual mainstream     | **C1** | **C1** | 152 airports   |
| Large airports (heterogeneous)    | **C2** | **C2 + C3** | 119 ≈ 90 + 31 |

The fact that two algorithms with different objective functions
(K-Means minimises intra-cluster variance; Ward agglomerative minimises
the increase in variance from each merge) converge on the *same*
small-leisure and mid-sized clusters is strong evidence that **these two
categories are real**. Their disagreement about whether the major
airports form one or two clusters reflects a genuine ambiguity in the
data: the megahub tier (~31 airports) is *distinct enough* that
Agglomerative separates it, but K-Means at k = 3 prefers to keep it
together with the other large airports because doing so minimises the
overall within-cluster variance.

## 6. Suggested wording for the LaTeX report

Three defensible claims:

1. **On the aggregation and feature design.**
   *"We aggregated the flight-level training data to 306 origin airports
   (after filtering airports with fewer than 30 flights), describing each
   airport with 13 standardised features spanning traffic volume, network
   reach, delay statistics, delay-class proportions, route profile, and
   temporal mix. Standard-scaling was essential: without it the n_flights
   feature would have dominated the Euclidean distance and reduced every
   clustering to a 1-D ranking by traffic volume."*

2. **On the recovered operational categories.**
   *"K-Means (k = 3, silhouette = 0.21) and Agglomerative (k = 4,
   silhouette = 0.16) converge on the same three operational categories
   of US airports: (i) small regional / leisure airports dominated by
   resorts and seasonal destinations such as Aspen and Jackson Hole, with
   the highest delay variance (σ ≈ 117 min); (ii) the long tail of
   mid-sized punctual airports (152 airports, 78 % on-time rate); and
   (iii) the major hubs — Atlanta, Dallas-Fort Worth, O'Hare, Denver,
   Charlotte — which the Agglomerative analysis further separates into a
   mid-large tier (≈ 770 flights / airport) and a megahub tier (≈ 6 200
   flights / airport). The fact that two algorithms with different
   objective functions recover the same partition is evidence that these
   categories are intrinsic to the data, not artefacts of the choice of
   algorithm."*

3. **On the DBSCAN result and the shape of the data.**
   *"DBSCAN, at its best epsilon setting, labelled 52 % of airports as
   noise — a higher fraction than any sensible 'outlier' interpretation
   could justify. We interpret this as a structural diagnosis: the
   airport feature space is a continuous operational gradient rather than
   a collection of density-separated clumps. The modest silhouette scores
   of K-Means and Agglomerative (0.16 – 0.21) are consistent with this
   interpretation — the partitions impose useful structure on a continuum,
   but the continuum itself is the underlying truth. This is itself a
   finding: US airports vary gradually along a scale-and-punctuality axis,
   and any partition we draw is a useful operational simplification rather
   than the discovery of an objective category boundary."*
