import pandas as pd
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif, chi2, VarianceThreshold
from sklearn.model_selection import train_test_split
from representation import DNA
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from mrmr import mrmr_classif

class SelectKFeature:
    def __init__(self, dataset, k_feature=100, kmer_size=2, domaine=None, sample_size=None, discriminative=None):
        """
        Initialize the SelectFeature class.

        Parameters:
        - dataset: DataFrame containing sequences and class labels (0 and 1).
        - k_feature: Number of top features to select.
        - kmer_size: k-mer size.
        - domaine: Domain of possible characters in sequences.
        - sample_size: Proportion of the dataset to include in the test split.
        """
        self.dataset = dataset
        self.k_feature = k_feature
        self.kmer_size = kmer_size
        self.domaine = domaine
        self.sample_size = sample_size
        self.discriminative = discriminative

    def sample_dataset(self):
        """
        Split the dataset into train and test sets with stratification.

        Returns:
        - test_set: DataFrame of the test set.
        """
        if self.sample_size and 0 < self.sample_size < 1:
            _, test_set = train_test_split(
                self.dataset,
                test_size=self.sample_size,
                stratify=self.dataset['class'],
                random_state=42
            )
        else:
            _, test_set = self.dataset, self.dataset
        
        if self.discriminative is not None:
            test_set = test_set[test_set['class'] == self.discriminative]
        
        test_set = test_set.reset_index(drop=True)
        return test_set

    def build_kmer_representation(self, sampled_dataset):
        """
        Build k-mer representation of the sampled dataset.
        
        Returns:
        - kmer_df: DataFrame of k-mer features.
        """
        X, y, features  = DNA.build_kmer_representation_v2(
            sampled_dataset,
            domaine=self.domaine,
            k=self.kmer_size,
            dtypes=['float16', 'int8'],
            asCudaDF=False
        )
        
        # Combine X and y into a single DataFrame
        kmer_df = pd.DataFrame(X, columns=features)
        kmer_df['class'] = y
        return kmer_df

    def select_features(self, kmer_df):
        """
        Select top x features using SelectKBest and f_classif.
        
        Parameters:
        - kmer_df: DataFrame of k-mer features.
        
        Returns:
        - selected_features: List of selected feature names.
        """
        X = kmer_df.drop(columns=['class'])
        y = kmer_df['class']

        selector = SelectKBest(score_func=f_classif, k=self.k_feature)
        selector.fit(X, y)

        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        return selected_features

    def select_features_rf(self, kmer_df):
        """
        Select top x features using Random Forest feature importance.
        
        Parameters:
        - kmer_df: DataFrame of k-mer features.
        
        Returns:
        - selected_features: List of selected feature names.
        """
        X = kmer_df.drop(columns=['class'])
        y = kmer_df['class']

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importances = rf.feature_importances_
        indices = importances.argsort()[::-1][:self.k_feature]
        selected_features = X.columns[indices].tolist()
        
        return selected_features

    def select_features_chi2(self, kmer_df):
        """
        Select top x features using Chi-squared (chi2) feature selection.
        
        Parameters:
        - kmer_df: DataFrame of k-mer features.
        
        Returns:
        - selected_features: List of selected feature names.
        """
        X = kmer_df.drop(columns=['class'])
        y = kmer_df['class']

        selector = SelectKBest(score_func=chi2, k=self.k_feature)
        selector.fit(X, y)

        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        return selected_features

    def select_features_variance_threshold(self, kmer_df, threshold=0.1):
        """
        Select features based on variance threshold.
        
        Parameters:
        - kmer_df: DataFrame of k-mer features.
        - threshold: Variance threshold value.
        
        Returns:
        - selected_features: List of selected feature names.
        """
        X = kmer_df.drop(columns=['class'])

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)

        selected_features = X.columns[selector.get_support()].tolist()
        return selected_features
        
    def select_features_lasso(self, kmer_df, alpha=0.1):
        """
        Select top x features using Lasso (L1 regularization).
        
        Parameters:
        - kmer_df: DataFrame of k-mer features.
        - alpha: Regularization strength.
        
        Returns:
        - selected_features: List of selected feature names.
        """
        X = kmer_df.drop(columns=['class'])
        y = kmer_df['class']

        lasso = Lasso(alpha=alpha, max_iter=1000)
        lasso.fit(X, y)

        # Get indices of non-zero coefficients (selected features)
        non_zero_indices = lasso.coef_ != 0
        selected_features = X.columns[non_zero_indices].tolist()
        
        return selected_features
        
    def get_feature_mask(self, method=['f_test']):
        """
        Get the feature mask of top x features based on the specified method.
        
        Parameters:
        - method: Feature selection method to use ('f_test', 'rf', 'chi2', 'variance-threshold').
        
        Returns:
        - feature_mask: List of selected feature names.
        """
        if method[0] == 'f_test':
            df_set = self.sample_dataset()
            kmer_df = self.build_kmer_representation(df_set)
            feature_mask = self.select_features(kmer_df)
        elif method[0] == 'rf':
            df_set = self.sample_dataset()
            kmer_df = self.build_kmer_representation(df_set)
            feature_mask = self.select_features_rf(kmer_df)
        elif method[0] == 'chi2':
            df_set = self.sample_dataset()
            kmer_df = self.build_kmer_representation(df_set)
            feature_mask = self.select_features_chi2(kmer_df)
        elif method[0] == 'variance-threshold':
            df_set = self.sample_dataset()
            kmer_df = self.build_kmer_representation(df_set)
            feature_mask = self.select_features_variance_threshold(kmer_df, method[1])
        elif method[0] == 'lasso':
            df_set = self.sample_dataset()
            kmer_df = self.build_kmer_representation(df_set)
            feature_mask = self.select_features_lasso(kmer_df, method[1])
        elif method[0] == 'mrmr':
            df_set = self.sample_dataset()
            kmer_df = self.build_kmer_representation(df_set)
            X = kmer_df.drop(columns=['class'])
            y = kmer_df['class']
            feature_mask = mrmr_classif(X=X, y=y, K=self.k_feature)
        else:
            raise ValueError("Unsupported method. Supported methods: 'f_test', 'rf', 'chi2', 'variance-threshold', 'lasso'.")
        
        return feature_mask