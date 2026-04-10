import pandas as pd

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by removing duplicates and handling missing values.
    Returns the cleaned DataFrame.
    """
    df_clean = df.drop_duplicates()
    df_clean = df_clean.dropna()
    return df_clean