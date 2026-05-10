#%% Libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import numpy as np

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor # predict a continuous scale of numbers, better than classifier who just understands labels
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
from itertools import combinations_with_replacement

from scipy.stats import f_oneway, kruskal, ttest_ind

import pickle
import umap
#pip install mrmr_selection
import math
import HypothesisTesting
import warnings
import DataCleaning
import DataLoader
import DataPreprocessing
import FeatureEngineering
import EDA
import FlightDimensionalityReduction
import DataVisualization
import Knn
import Ensemble
import DeepLearning
warnings.filterwarnings("ignore")


#%% DataLoader
data_loader = DataLoader("data/flights_sample_3m.csv")
# Shows the before and after of the Data Analysis
print("\n\n--- 1. Raw Data Loaded ---")
print("Training data shape:", data_loader.data_train.shape)
print("Training labels shape:", data_loader.labels_train.shape)
print("Testing data shape:", data_loader.data_test.shape)
print("Testing labels shape:", data_loader.labels_test.shape)
#%% DataCleaning
data_cleaner = DataCleaning(data_loader)
data_cleaner.remove_leakage()
data_cleaner.remove_duplicates()
data_cleaner.handle_missing_values()
data_cleaner.remove_outliers()
print("\n\n--- 2. After Data Cleaning ---")
print("Training data shape:", data_loader.data_train.shape)
print("Training labels shape:", data_loader.labels_train.shape)
print("Testing data shape:", data_loader.data_test.shape)
print("Testing labels shape:", data_loader.labels_test.shape)
#%%FeatureEngineering
# 1. Aplicar a função aos dados de Treino e Teste
print("A processar Feature Engineering...")
fe = FeatureEngineering()
data_loader.data_train = fe.perform_feature_engineering(
    data_loader.data_train,
    target=data_loader.labels_train
)
data_loader.data_test = fe.perform_feature_engineering(
    data_loader.data_test
)
# 2. Definir a lista das colunas novas para filtrar a tabela
novas_colunas = [
    'FL_DATE','MONTH','IS_HOLIDAY_MONTH','DAY_OF_WEEK','IS_WEEKEND','SEASON','CRS_DEP_HOUR','TIME_OF_DAY','FLIGHT_TYPE','ROUTE','CRS_ARR_HOUR','PLANNED_SPEED','AVG_DELAY_PER_HOUR'
]
# 3. Visualizar o resultado!
print("\n--- Novas Features (Primeiras 5 linhas do Treino) ---")
display(data_loader.data_train[novas_colunas].head())
#%% DataPreprocessing
# 1. include the newly engineered features
num_cols = ['CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE', 'CRS_DEP_HOUR', 'CRS_ARR_HOUR', 'PLANNED_SPEED', 'AVG_DELAY_PER_HOUR']
cat_cols = ['AIRLINE_CODE', 'ORIGIN', 'DEST', 'MONTH', 'DAY_OF_WEEK', 'IS_WEEKEND', 'SEASON', 'IS_HOLIDAY_MONTH', 'TIME_OF_DAY', 'FLIGHT_TYPE', 'ROUTE']

# 2. CHECKPOINT 1: Cleaned Data (Before encoding - for EDA & Hypothesis Testing)
import pickle
with open('data_loader_cleaned.pkl', 'wb') as f:
    pickle.dump(data_loader, f)
print("Saved Checkpoint 1: 'data_loader_cleaned.pkl' (Ready for EDA and Hypothesis Testing)")

# 3. APPLY PREPROCESSING (Scaling and One-Hot Encoding)
data_preprocessing = DataPreprocessing(data_loader, num_cols, cat_cols)
print("\n--- 3. After Data Preprocessing (Scaled & Encoded) ---")

# 4. SAVE CHECKPOINT 2: Preprocessed Data (After encoding - for Machine Learning)
with open('data_loader_preprocessed.pkl', 'wb') as f:
    pickle.dump(data_loader, f)
print("Saved Checkpoint 2: 'data_loader_preprocessed.pkl' (Ready for the ML Model)")

# 5. Deserialize the preprocessed data to verify it loaded correctly
with open('data_loader_preprocessed.pkl', 'rb') as f:
    data_loader_loaded = pickle.load(f)

print("\n\nDeserialized Preprocessed Data Check:")
print("Training data shape:", data_loader_loaded.data_train.shape)
print("Training labels shape:", data_loader_loaded.labels_train.shape)
print("Testing data shape:", data_loader_loaded.data_test.shape)
print("Testing labels shape:", data_loader_loaded.labels_test.shape)

#%% EDA (Exploratory Data Analysis)
eda = EDA(data_loader,num_cols,cat_cols)
eda.perform_eda()
#%% DataVisualization(Performs data visualization methods)
# 1. Combine your preprocessed columns so the algorithm knows what to look at
features_to_use = num_cols + cat_cols
datavisualization = DataVisualization(data_loader, features_to_use)
datavisualization.perform_visualization()
#%% Dimension Reduction

# 2. Initialize the class (it automatically samples the data)
dr = FlightDimensionalityReduction(data_loader, features_to_use)

# 3. Compute and plot PCA (Linear)
pca_proj = dr.compute_pca()
dr.plot_projection(pca_proj, 'PCA Projection (Linear)')

# 4. Compute and plot UMAP (Non-Linear)
umap_proj = dr.compute_umap()
dr.plot_projection(umap_proj, 'UMAP Projection (Non-Linear)')
features_to_investigate = [
    'DISTANCE', 'AIRLINE_CODE', 'ORIGIN', 'TIME_OF_DAY',
    'SEASON', 'FLIGHT_TYPE', 'DAY_OF_WEEK', 'MONTH', 'IS_HOLIDAY_MONTH','PLANNED_SPEED','ROUTE', 'AVG_DELAY_PER_HOUR','DEST', 'CRS_ELAPSED_TIME'
]
dr.analyze_umap_clusters(umap_proj,features_to_investigate)

tsne_proj = dr.compute_tsne()
dr.plot_projection(tsne_proj, 'TSNE Projection (Non-Linear)')

#%%HypothesisTesting

# 1. Load the Cleaned Data (Before it was One-Hot Encoded)
with open('data_loader_cleaned.pkl', 'rb') as f:
    data_loader_for_testing = pickle.load(f)

# 2. Run the Tests!
ht = HypothesisTesting(data_loader_for_testing)
ht.run_all_tests()
#%%Knn
#using the preprocessed because the data need to be all the same so theres no overpower difference. All the text is in 1s and 0s.
with open('data_loader_preprocessed.pkl', 'rb') as f:
    data_loader = pickle.load(f)

# 2. Extract your X (Features) - These are already scaled!
X_train_scaled = data_loader.data_train.values # .values converts Pandas DataFrame to NumPy Array
X_test_scaled = data_loader.data_test.values

def categorize_delay(delay):
    if delay < 15:
        return 0  # On-time
    elif 15 <= delay <= 30:
        return 1  # Short delay
    else:
        return 2  # Long delay

# Assuming data_loader is loaded from your checkpoint
y_train_class = data_loader.labels_train.apply(categorize_delay)
y_test_class = data_loader.labels_test.apply(categorize_delay)

# Now you can train your custom KNN!
knn = Knn(k=5)
knn.fit(X_train_scaled, y_train_class)

# Remember to test on a small sample first! (e.g., 5000 rows)
accuracy = knn.score(X_test_scaled[:5000], y_test_class[:5000])
print(f"Custom KNN Accuracy: {accuracy:.4f}")
#%% Ensemble
# Bagging (Random Forest) + Boosting (Hist Gradient Boosting) on the multiclass delay target
with open('data_loader_preprocessed.pkl', 'rb') as f:
    data_loader_ensemble = pickle.load(f)

exp1 = Ensemble(data_loader_ensemble, train_sample_size=200000, test_sample_size=50000, balance_training=False)
exp1.run_all()
exp2 = Ensemble(data_loader_ensemble, train_sample_size=200000, test_sample_size=50000, balance_training=True)
exp2.run_all()

#%% Deep Learning (MLP via TensorFlow/Keras)
# MLP for the multiclass delay target — same train/test sizes as Ensemble for a fair comparison.
with open('data_loader_preprocessed.pkl', 'rb') as f:
    data_loader_dl = pickle.load(f)

# --- Experiment 1: classification (3 classes) with sklearn class_weight + macro-F1 callback. ---
# CHAMPION MLP. The custom macro-F1 callback prevents the degenerate-majority
# collapse that the default EarlyStopping(val_loss) would silently produce.
print("=" * 60)
print("DEEP LEARNING — Experiment 1: Classification (sklearn-balanced + macro-F1 callback)")
print("=" * 60)
dl_clf = DeepLearning(
    data_loader_dl,
    task='classification',
    train_sample_size=200000,
    test_sample_size=50000,
    balance_training=False,
)
dl_clf.run_all(hidden_units=(128, 64, 32), dropout_rate=0.3,
               learning_rate=1e-3, epochs=80, batch_size=512, patience=10)

# --- Experiment 1b: stress-test of "more fixes is better". ---
# Drops BatchNorm, uses stronger manual weights, and adds log-prior bias init.
# Empirically this OVER-CORRECTS and the model re-collapses to majority,
# documenting that the macro-F1 callback alone (Exp 1) was the necessary AND
# sufficient intervention.
print("=" * 60)
print("DEEP LEARNING — Experiment 1b: Over-correction stress-test")
print("=" * 60)
dl_clf_fix = DeepLearning(
    data_loader_dl,
    task='classification',
    train_sample_size=200000,
    test_sample_size=50000,
    balance_training=False,
)
dl_clf_fix.run_all(
    hidden_units=(128, 64, 32),
    dropout_rate=0.3,
    learning_rate=1e-3,
    epochs=80,
    batch_size=512,
    patience=10,
    use_batch_norm=False,
    class_weight_override={0: 1.0, 1: 5.0, 2: 2.5},
)

# --- Experiment 2: classification on a balanced (undersampled) training set ---
print("=" * 60)
print("DEEP LEARNING — Experiment 2: Classification (balanced training)")
print("=" * 60)
dl_clf_bal = DeepLearning(
    data_loader_dl,
    task='classification',
    train_sample_size=200000,
    test_sample_size=50000,
    balance_training=True,
)
dl_clf_bal.run_all(hidden_units=(128, 64, 32), dropout_rate=0.3,
                   learning_rate=1e-3, epochs=80, batch_size=512, patience=10)

# --- Experiment 3: regression (predict ARR_DELAY in minutes) ---
print("=" * 60)
print("DEEP LEARNING — Experiment 3: Regression on ARR_DELAY")
print("=" * 60)
dl_reg = DeepLearning(
    data_loader_dl,
    task='regression',
    train_sample_size=200000,
    test_sample_size=50000,
)
dl_reg.run_all(hidden_units=(128, 64, 32), dropout_rate=0.3,
               learning_rate=1e-3, epochs=80, batch_size=512, patience=10)

#%% Hierarchical Deep Learning (Classify then Regress)
# Two-stage cascade: Stage 1 = binary classifier (On-time vs Delayed),
# Stage 2 = regression on minute-level delay, trained ONLY on delayed flights.
# Rationale: Flat 3-class MLP and flat regression both struggle because the
# pre-departure features have moderate signal for "is it late?" and very weak
# signal for "by how much?". Separating the two questions lets each stage
# specialize.
from DeepLearning.HierarchicalDeepLearning import HierarchicalDeepLearning

with open('data_loader_preprocessed.pkl', 'rb') as f:
    data_loader_hdl = pickle.load(f)

print("=" * 60)
print("HIERARCHICAL DL — Stage 1 (binary) + Stage 2 (regression)")
print("=" * 60)
hdl = HierarchicalDeepLearning(
    data_loader_hdl,
    train_sample_size=200000,
    test_sample_size=50000,
    delay_threshold=15,
)
hdl.run_all(
    hidden_units=(128, 64, 32),
    dropout_rate=0.3,
    learning_rate=1e-3,
    epochs=80,
    batch_size=512,
    patience=10,
    threshold=0.5,
)

#%% Clustering — airport operational profiles
# Aggregates flights to one row per ORIGIN airport, then applies three
# clustering algorithms (K-Means, DBSCAN, Agglomerative) over varying
# n_clusters / eps to identify operational patterns.
# Uses the CLEANED data (before one-hot) so ORIGIN / DEST / AIRLINE_CODE
# are still raw strings the aggregator can group on.
from Clustering.Clustering import Clustering

with open('data_loader_cleaned.pkl', 'rb') as f:
    data_loader_cluster = pickle.load(f)

print("=" * 60)
print("CLUSTERING — Airports by operational profile")
print("=" * 60)
clust = Clustering(
    data_loader_cluster,
    entity='airport',
    sample_size=300000,
    min_flights=30,
)
clust.run_all(
    k_values=range(2, 9),
    eps_values=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
    agglo_k=4,
)





































