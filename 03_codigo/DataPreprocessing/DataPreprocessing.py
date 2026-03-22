import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

class DataPreprocessing:
    """
    Class responsible for encoding and normalizing the loaded dataset. Transforms the categorical data in numerical
    Attributes:
        data_loader (DataLoader): An object of the DataLoader class.
        numerical_cols (list): list of numerical column names
        categorical_cols (list): list of categorical column names
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