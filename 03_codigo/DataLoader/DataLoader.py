import pandas as pd
from sklearn.model_selection import train_test_split


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
        :param filename: filename of the Flight dataset.
        :param test_size: The size of the test split.
        :param random_state: The random state for reproducibility.
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
        Loads the dataset from the specified filename,
        splits it into training and testing sets using train_test_split(),
        and assigns the data and labels to the appropriate attributes.
        """
        try:
            # Load the dataset
            df = pd.read_csv(self.filename, low_memory=False)
            print(f"Original shape: {df.shape}")

            # "Cancelled or diverted flights do not have meaningful arrival delay values"
            df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]
            print(f"Shape after removing cancelled/diverted: {df.shape}")

            #Target (y) and Features (X)
            y = df['ARR_DELAY']
            X = df.drop(columns=['ARR_DELAY'])

            # Split the data
            self.data_train, self.data_test, self.labels_train, self.labels_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            print("Data loaded and split successfully.")

        except FileNotFoundError:
            print("File not found. Please check the file path.")