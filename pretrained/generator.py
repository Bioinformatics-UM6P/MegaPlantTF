import pandas as pd

class DataFrameProcessor:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.sort_index(axis=1)
    
    def fit_mask(self, feature_array):
        df_copy = self.dataframe.copy()
        existing_columns = df_copy.columns.intersection(feature_array)
        
        # Drop columns not in the feature_array
        df_copy = df_copy[existing_columns]
        
        # Create a new DataFrame with the feature_array as columns, initialize with zeros
        new_df = pd.DataFrame(0, index=df_copy.index, columns=feature_array)
        for col in existing_columns:
            new_df[col] = df_copy[col]
        
        return new_df

    def get(self):
        return self.dataframe