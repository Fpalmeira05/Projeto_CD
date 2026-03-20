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

import math

import warnings
warnings.filterwarnings("ignore")

#%%Pre-processing
class DataLoader:
    """
    Class responsible for loading the Flight dataset and splitting it.
    Attributes:
        filename : The filename of the Flight dataset.
        test_size : The size of the test split.
        random_state : The random state for reproducibility.
    Attributes (after loading the data):
        data_train (DataFrame): The training data features.
        labels_train (Series): The training data labels.
        data_test (DataFrame): The testing data features.
        labels_test (Series): The testing data labels.
    Methods:
        _load_data(): Loads the dataset, splits it into training and testing sets,
                      and assigns the data and labels to the appropriate attributes.
    """

    def __init__(self, filename, test_size=0.2, random_state=42):
        """
        Initializes the DataLoader with the filename of the dataset,
        the proportion of data to include in the test split,
        and the random state for reproducibility.
        :param filename:
        :param test_size:
        :param random_state:
        """
        self.filename = filename
        self.test_size = test_size
        self.random_state = random_state
        self.data_train = None
        self.labels_train = None
        self.data_test = None
        self.labels_test = None

        # Load data immediately
        self._load_data()

    def _load_data(self):
        """
        Loads the dataset, validates it, splits into train/test,
        and assigns the data and labels to the appropriate attributes.
        """
        import os

        # 1. Validate filename
        if not isinstance(self.filename, str) or not self.filename.strip():
            raise ValueError("Filename must be a non-empty string.")

        # 2. Check file extension
        if not self.filename.lower().endswith('.csv'):
            raise ValueError(f"Expected a .csv file, got: '{self.filename}'")

        # 3. Check if file exists
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"File not found: '{self.filename}'. Please check the file path.")

        # 4. Check if file is empty (0 bytes)
        if os.path.getsize(self.filename) == 0:
            raise ValueError(f"File is empty: '{self.filename}'")

        # 5. Validate split parameters
        if not (0 < self.test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1 (exclusive), got: {self.test_size}")

        if not isinstance(self.random_state, int) or self.random_state < 0:
            raise ValueError(f"random_state must be a non-negative integer, got: {self.random_state}")

        # 6. Read the CSV
        try:
            df = pd.read_csv(self.filename, low_memory=False)
        except pd.errors.EmptyDataError:
            raise ValueError(f"File has no data to parse: '{self.filename}'")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Encoding error reading file (try specifying encoding): {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error reading file: {e}")

        print(f"Original shape: {df.shape}")

        # 7. Check if DataFrame is empty
        if df.empty:
            raise ValueError("The loaded DataFrame is empty (0 rows).")


        # 8. Check for required columns
        required_columns = ['ARR_DELAY', 'CANCELLED', 'DIVERTED']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}. "
                           f"Available columns: {list(df.columns)}")

        # 9. Validate data types of key columns
        for col in ['ARR_DELAY', 'CANCELLED', 'DIVERTED']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' must be numeric, got dtype: {df[col].dtype}")

        # 10. Check for valid CANCELLED/DIVERTED values (should be 0 or 1)
        for col in ['CANCELLED', 'DIVERTED']:
            unique_vals = df[col].dropna().unique()
            invalid_vals = [v for v in unique_vals if v not in [0, 1]]
            if invalid_vals:
                print(f"Warning: Column '{col}' contains unexpected values: {invalid_vals}. "
                      f"Expected only 0 and 1.")

        # 11. Filter out cancelled and diverted flights
        df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]
        print(f"Shape after removing cancelled/diverted: {df.shape}")

        # 12. Check if any rows remain after filtering
        if df.empty:
            raise ValueError("No rows remaining after filtering cancelled/diverted flights. "
                             "All flights are either cancelled or diverted.")

        # 13. Check for target variable issues
        if df['ARR_DELAY'].isnull().all():
            raise ValueError("Target column 'ARR_DELAY' has no valid (non-null) values.")

        null_target_count = df['ARR_DELAY'].isnull().sum()
        if null_target_count > 0:
            print(f"Warning: 'ARR_DELAY' has {null_target_count} null values "
                  f"({null_target_count / len(df) * 100:.2f}%). These rows will be dropped.")
            df = df.dropna(subset=['ARR_DELAY'])

        # 14. Check if enough rows exist for splitting
        min_rows_needed = max(int(1 / self.test_size), int(1 / (1 - self.test_size)), 2)
        if len(df) < min_rows_needed:
            raise ValueError(f"Not enough rows ({len(df)}) to split with test_size={self.test_size}. "
                             f"Need at least {min_rows_needed} rows.")

        # 15. Separate target and features
        y = df['ARR_DELAY']
        X = df.drop(columns=['ARR_DELAY'])

        # 16. Check if there are any features left
        if X.shape[1] == 0:
            raise ValueError("No feature columns remain after removing 'ARR_DELAY'.")

        # 17. Report duplicate info
        n_duplicates = X.duplicated().sum()
        if n_duplicates > 0:
            print(f"Info: Dataset contains {n_duplicates} duplicate rows "
                  f"({n_duplicates / len(X) * 100:.2f}%). Consider removing them later.")

        # 18. Report missing values info
        total_missing = X.isnull().sum().sum()
        if total_missing > 0:
            cols_with_missing = X.columns[X.isnull().any()].tolist()
            print(f"Info: {total_missing} missing values found across columns: {cols_with_missing}")

        # 19. Perform the train/test split
        try:
            self.data_train, self.data_test, self.labels_train, self.labels_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        except Exception as e:
            raise RuntimeError(f"Error during train/test split: {e}")

        # 20. Final validation
        assert len(self.data_train) > 0, "Training set is empty after split."
        assert len(self.data_test) > 0, "Test set is empty after split."
        assert len(self.data_train) == len(self.labels_train), "Train data/labels size mismatch."
        assert len(self.data_test) == len(self.labels_test), "Test data/labels size mismatch."

        print(f"Data loaded and split successfully.")
        print(f"  Train: {self.data_train.shape[0]} rows, {self.data_train.shape[1]} features")
        print(f"  Test:  {self.data_test.shape[0]} rows, {self.data_test.shape[1]} features")


class DataPreprocessing:
    """
    Class responsible for encoding and normalizing the loaded dataset. Transforms the categorical data in numerical

    Methods:
        _encode_categorical(): Normalizes all features that are categorical into numerical
        _normalize_features(): Normalizes numerical features using StandardScaler for Numerical and MinMaxScaler for categorical
    """

    def __init__(self, data_loader, numerical_cols, categorical_cols):
        """
        Initializes the DataPreprocessing class.
        Now takes explicit lists of numerical and categorical column names.
        """
        self.data_loader = data_loader
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

        # 1. First encode the text to numbers
        self._encode_categorical()

        # 2. Then normalize the scales
        self._normalize_features()

    def _encode_categorical(self):
        """
        Translates text categories (like airport codes) into numerical labels.
        """
        try:
            for col in self.categorical_cols:
                le = LabelEncoder()

                # Fit the encoder on BOTH train and test data to learn all possible categories
                combined_data = pd.concat([self.data_loader.data_train[col], self.data_loader.data_test[col]])
                le.fit(combined_data)

                # Transform both sets using the comprehensively fitted encoder
                self.data_loader.data_train[col] = le.transform(self.data_loader.data_train[col])
                self.data_loader.data_test[col] = le.transform(self.data_loader.data_test[col])

            print(f"Categorical features {self.categorical_cols} encoded successfully.")
        except Exception as e:
            print(f"Error encoding categorical features: {e}")

    def _normalize_features(self):
        """
        Normalizes numerical features using StandardScaler and categorical using MinMaxScaler.
        """
        try:
            # Check if data exists
            if self.data_loader.data_train is None or self.data_loader.data_test is None:
                raise ValueError("Data has not been loaded yet.")

            # Normalize numerical features using StandardScaler
            if self.numerical_cols:
                std_scaler = StandardScaler()
                self.data_loader.data_train[self.numerical_cols] = std_scaler.fit_transform(
                    self.data_loader.data_train[self.numerical_cols])
                self.data_loader.data_test[self.numerical_cols] = std_scaler.transform(
                    self.data_loader.data_test[self.numerical_cols])

            # Normalize encoded categorical features using MinMaxScaler
            if self.categorical_cols:
                minmax_scaler = MinMaxScaler()
                self.data_loader.data_train[self.categorical_cols] = minmax_scaler.fit_transform(
                    self.data_loader.data_train[self.categorical_cols])
                self.data_loader.data_test[self.categorical_cols] = minmax_scaler.transform(
                    self.data_loader.data_test[self.categorical_cols])

            print("Features normalized successfully.")

        except ValueError as ve:
            print(f"Error normalizing: {ve}")



class FeatureEngineering:
    def perform_feature_engineering(self, df):
            """
            Create 10 new features from the existing variables(no data leakage)
            :param self:
            :param df:
            :return:
            """

            df_fe = df.copy()

            # Garantir que a data está no formato correto de DateTime
            df_fe['FL_DATE'] = pd.to_datetime(df_fe['FL_DATE'])

            # 1. Mês do ano (1 a 12)
            df_fe['MONTH'] = df_fe['FL_DATE'].dt.month

            #2. Dia da semana(0 a 5 , 0 = Segunda  e 6 = Domingo)
            df_fe['DAY_OF_WEEK'] = df_fe['FL_DATE'].dt.dayofweek

            #3. Verifica se e fim de semana(sim = 1 nao = 0)
            df_fe['IS_WEEKEND'] = np.where(df_fe['DAY_OF_WEEK'] >= 5, 1, 0)

            # 4. Estação do ano (1=Inverno, 2=Primavera, 3=Verão, 4=Outono)
            df_fe['SEASON'] = df_fe['MONTH'].apply(
                lambda x: 1 if x in [12, 1, 2] else (2 if x in [3, 4, 5] else (3 if x in [6, 7, 8] else 4))
            )

            # 5. Verifica se esta na altura de ferias(julho = 7,agosto = 8 e dezembro = 12), 1 = sim 0 = nao
            df_fe['IS_HOLIDAY_MONTH'] = np.where(df_fe['MONTH'].isin([7, 8, 12]), 1, 0)

            # 6. Hora Planeada de Partida
            # O CRS_DEP_TIME vem frequentemente no formato hhmm (ex: 1530 para 15:30).
            # A divisão inteira por 100 extrai apenas a hora (15).
            df_fe['CRS_DEP_HOUR'] = df_fe['CRS_DEP_TIME'] // 100
            # Correção para casos raros onde a hora passa as 24h
            df_fe['CRS_DEP_HOUR'] = df_fe['CRS_DEP_HOUR'].replace(24,0)

            #7. Periodo do Dia (0 = Madrugada, 1 = Manha, 2 = Tarde, 3 = Noite)
            bins = [-1,5,11,18,25] #(0 as 5 horas = Madrugada, 5 as 11 = Manha, 11 as 18 = Tarde, 18 as 23 = Noite)
            labels_num = [0, 1, 2, 3]
            df_fe['TIME_OF_DAY'] = pd.cut(df_fe['CRS_DEP_HOUR'], bins=bins, labels=labels_num)

            # 8. Rota (Origem -> Destino)
            # O modelo pode descobrir que a rota "JFK_LAX" se atrasa mais que outras.
            df_fe['ROUTE'] = df_fe['ORIGIN'] + "_" + df_fe['DEST']

            # 10. Tipo de Voo (Curto, Médio ou Longo Curso)
            # Curto < 500 milhas | Médio 500 a 1500 milhas | Longo > 1500 milhas
            bins_dist = [-1, 500, 1500, df_fe['DISTANCE'].max() + 1]
            labels_dist = [0, 1, 2]  # 0: Curto, 1: Médio, 2: Longo
            df_fe['FLIGHT_TYPE'] = pd.cut(df_fe['DISTANCE'], bins=bins_dist, labels=labels_dist)
            df_fe['FLIGHT_TYPE'] = df_fe['FLIGHT_TYPE'].astype(float)

            # 2. Hora Planeada de Chegada
            df_fe['CRS_ARR_HOUR'] = df_fe['CRS_ARR_TIME'] // 100
            df_fe['CRS_ARR_HOUR'] = df_fe['CRS_ARR_HOUR'].replace(24, 0)  # Corrige as 24h para 0h

            # 3. Velocidade Planeada (Milhas por Minuto)
            # Somamos 0.001 para evitar um erro fatal caso algum tempo seja 0
            df_fe['PLANNED_SPEED'] = df_fe['DISTANCE'] / (df_fe['CRS_ELAPSED_TIME'] + 0.001)





            return df_fe




class DataCleaning:
    """
    Class for cleaning operations.
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def remove_duplicates(self):
        """
        Remove duplicate rows from the train dataset.
        """
        try:
            # Check if data and labels are not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")
            if self.data_loader.labels_train is None:
                raise ValueError("Labels have not been loaded yet.")

            # Remove duplicate rows from training data (do not apply to test data)
            self.data_loader.data_train.drop_duplicates(inplace=True)
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]

            print("Duplicate rows removed from training data.")

        except ValueError as ve:
            print("Error:", ve)

    def handle_missing_values(self, strategy='drop'):
        """
        Handle missing values using the specified strategy.

        Parameters:
            strategy (str): The strategy to handle missing values ('mean', 'median', 'most_frequent', or a constant value).
        """
        try:
            # Check if data is not None
            if self.data_loader.data_train is None or self.data_loader.data_test is None:
                raise ValueError("Data has not been loaded yet.")

            # Check if there are missing values
            if self.data_loader.data_train.isnull().sum().sum() == 0 and self.data_loader.data_test.isnull().sum().sum() == 0:
                print("No missing values found in the data.")
                return

            # Handle missing values based on the specified strategy
            if strategy == 'mean':
                self.data_loader.data_train.fillna(self.data_loader.data_train.mean(), inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.mean(), inplace=True)
            elif strategy == 'median':
                self.data_loader.data_train.fillna(self.data_loader.data_train.median(), inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.median(), inplace=True)
            elif strategy == 'most_frequent':
                self.data_loader.data_train.fillna(self.data_loader.data_train.mode().iloc[0], inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.mode().iloc[0], inplace=True)
            elif strategy == 'fill_nan':
                self.data_loader.data_train.fillna(strategy, inplace=True)
                self.data_loader.data_test.fillna(strategy, inplace=True)
            elif strategy == 'drop':
                self.data_loader.data_train = self.data_loader.data_train.dropna(axis=0)
                self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]
                self.data_loader.data_test = self.data_loader.data_test.dropna(axis=0)
                self.data_loader.labels_test = self.data_loader.labels_test[self.data_loader.data_test.index]

            else:
                raise ValueError("Invalid strategy.")
            print("Missing values handled using strategy:", strategy)

        except ValueError as ve:
            print("Error:", ve)

    def _detect_outliers(self, threshold=4):
        """
        Detect outliers in numerical features using z-score method.

        Parameters:
            threshold (float): The threshold value for determining outliers.

        Returns:
            outliers (DataFrame): DataFrame containing the outliers.
        """
        try:
            # Check if test data is not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")

            # Identify numerical features
            numerical_features = self.data_loader.data_train.select_dtypes(include=['number'])

            # Calculate z-scores for numerical features
            z_scores = (numerical_features - numerical_features.mean()) / numerical_features.std()

            # Find outliers based on threshold
            outliers = self.data_loader.data_train[(z_scores.abs() > threshold).any(axis=1)]

            return outliers

        except ValueError as ve:
            print("Error:", ve)

    def remove_outliers(self, threshold=2):
        """
        Remove outliers from the dataset using z-score method.

        Parameters:
            threshold (float): The threshold value for determining outliers.
        """
        try:
            # Check if data_loader.data is not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")

            # Detect outliers
            outliers = self._detect_outliers(threshold)

            # Remove outliers from the dataset
            self.data_loader.data_train = self.data_loader.data_train.drop(outliers.index)
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]

            print("Outliers removed from the dataset.")

        except ValueError as ve:
            print("Error:", ve)

    def remove_leakage(self):
        """
        Removes columns that cause data leakage as defined in the project specs.
        """
        # List from Project Specification [cite: 184-196]
        leakage_cols = [
            'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON',
            'TAXI_IN', 'ARR_TIME', 'DEP_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME',
            'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
            'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT',
            'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED'
        ]

        # Only drop columns that actually exist in the dataframe
        cols_to_drop = [col for col in leakage_cols if col in self.data_loader.data_train.columns]

        if cols_to_drop:
            self.data_loader.data_train.drop(columns=cols_to_drop, inplace=True)
            self.data_loader.data_test.drop(columns=cols_to_drop, inplace=True)
            print(f"Removed {len(cols_to_drop)} leakage columns.")



data_loader = DataLoader("data/flights_sample_3m.csv")
# Shows the before and after of the Data Analysis
print("\n\nBefore data preprocessing")
print("Training data shape:", data_loader.data_train.shape)
print("Training labels shape:", data_loader.labels_train.shape)
print("Testing data shape:", data_loader.data_test.shape)
print("Testing labels shape:", data_loader.labels_test.shape)

data_cleaner = DataCleaning(data_loader)
data_cleaner.remove_leakage()
data_cleaner.remove_duplicates()
data_cleaner.handle_missing_values()
data_cleaner.remove_outliers()
print("\n\nAfter data preprocessing")
print("Training data shape:", data_loader.data_train.shape)
print("Training labels shape:", data_loader.labels_train.shape)
print("Testing data shape:", data_loader.data_test.shape)
print("Testing labels shape:", data_loader.labels_test.shape)

num_cols = ['CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE']
cat_cols = ['AIRLINE_CODE', 'ORIGIN', 'DEST']
data_preprocessing = DataPreprocessing(data_loader,num_cols,cat_cols)
print("After data preprocessing:", data_preprocessing)

# Serialize data_loader object to save a copy of the cleaned data
with open('data_loader.pkl', 'wb') as f:
    pickle.dump(data_loader, f)

# Deserialize data_loader object to restore the cleaned data
with open('data_loader.pkl', 'rb') as f:
    data_loader_loaded = pickle.load(f)
print("\n\nDeserialized data")
print("Training data shape:", data_loader_loaded.data_train.shape)
print("Training labels shape:", data_loader_loaded.labels_train.shape)
print("Testing data shape:", data_loader_loaded.data_test.shape)
print("Testing labels shape:", data_loader_loaded.labels_test.shape)

#%% EDA (Exploratory Data Analysis)

class EDA:
    """
    A class responsible for exploratory data analysis.

    Attributes:
        data_loader (DataLoader): Object containing the dataset.

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
        self.plot_distributions()

        # Correlation analysis
        print("\nCorrelation Analysis:")
        self.plot_correlation_heatmap()

        # Feature Importance analysis
        print("\nFeature Importance Analysis:")
        self.plot_feature_importance()

    def plot_distributions(self, sample_size = 100000, cols = 3):
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

    def plot_correlation_heatmap(self):
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

    def plot_feature_importance(self, n_estimators=10, n_repeats=2, sample_size=50000):
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



eda = EDA(data_loader,num_cols,cat_cols)
eda.perform_eda()
#%% Dimension Reduction

class FlightDimensionalityReduction:
    """
    Adapted DimensionalityReduction class for the Flight Delay dataset.
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

# 1. Combine your preprocessed columns so the algorithm knows what to look at
features_to_use = num_cols + cat_cols

# 2. Initialize the class (it automatically samples the data)
dr = FlightDimensionalityReduction(data_loader, features_to_use)

# 3. Compute and plot PCA (Linear)
pca_proj = dr.compute_pca()
dr.plot_projection(pca_proj, 'PCA Projection (Linear)')

# 4. Compute and plot UMAP (Non-Linear)
umap_proj = dr.compute_umap()
dr.plot_projection(umap_proj, 'UMAP Projection (Non-Linear)')

tsne_proj = dr.compute_tsne()
dr.plot_projection(tsne_proj, 'TSNE Projection (Non-Linear)')







































