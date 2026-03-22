import pandas as pd
from scipy.stats import f_oneway, kruskal, ttest_ind
import itertools


class HypothesisTesting:
    """
    A class to perform statistical Hypothesis Testing on categorical features
    against a continuous target variable (ARR_DELAY).

    Attributes:
        data_loader(DataLoader): the DataLoader class to load the data.
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        # We drop NaNs from the target to ensure the math doesn't crash
        self.target = self.data_loader.labels_train.dropna()
        self.data = self.data_loader.data_train.loc[self.target.index]

    def _perform_t_test(self, feature):
        """
        T-Test for binary features (Exactly 2 categories, like 0 and 1).
        """
        classes = self.data[feature].dropna().unique()
        if len(classes) == 2:
            group1 = self.target[self.data[feature] == classes[0]]
            group2 = self.target[self.data[feature] == classes[1]]

            t_statistic, p_value = ttest_ind(group1, group2, equal_var=False)
            significant = p_value < 0.05
            print(f"Feature: {feature: <18} | p-value: {p_value:.4e} | Significant: {significant}")
        else:
            print(f"Feature: {feature: <18} | Skipped (Not binary)")

    def _perform_kruskal_test(self, feature):
        """
        Kruskal-Wallis test for features with 3 or more categories.
        (Statistically better than ANOVA for data with outliers like flight delays).
        """
        classes = self.data[feature].dropna().unique()
        if len(classes) > 2:
            groups = [self.target[self.data[feature] == c] for c in classes]
            h_statistic, p_value = kruskal(*groups)
            significant = p_value < 0.05
            print(f"Feature: {feature: <18} | p-value: {p_value:.4e} | Significant: {significant}")
        else:
            print(f"Feature: {feature: <18} | Skipped (Needs 3+ categories)")

    def run_all_tests(self):
        """
        Runs the appropriate tests for our engineered categorical features.
        """
        print("--- Hypothesis Testing Results (Target: ARR_DELAY) ---\n")

        binary_features = ['IS_WEEKEND', 'IS_HOLIDAY_MONTH']
        print(">>> T-TESTS (Comparing 2 Groups) <<<")
        for feature in binary_features:
            if feature in self.data.columns:
                self._perform_t_test(feature)

        multiclass_features = ['TIME_OF_DAY', 'SEASON', 'FLIGHT_TYPE', 'DAY_OF_WEEK', 'MONTH']
        print("\n>>> KRUSKAL-WALLIS TESTS (Comparing 3+ Groups) <<<")
        for feature in multiclass_features:
            if feature in self.data.columns:
                self._perform_kruskal_test(feature)