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






































