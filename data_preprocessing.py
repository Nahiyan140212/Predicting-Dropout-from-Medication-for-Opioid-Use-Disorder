"""
file: data_preprocessing.py

This script contains functions for data preprocessing tasks including:
- Handling missing values
- Standardizing race and ethnicity
- Merging race and ethnicity into a single column
- Creating failure rates
- Dropping specific columns
- One-hot encoding categorical columns
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import logging


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def handle_missing_values(df):
    """
    Handles missing values in a DataFrame:
      1. Fills categorical missing values with 'Unknown'.
      2. Imputes numerical missing values using the median.
      3. Applies KNN imputation for selected numerical columns.
      4. Fills missing 'CountOfOutpatVisit' values using the mean for YearOfInitiation (2018-2024).

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with missing values handled.
    """

    # 1. Handle categorical missing values
    categorical_cols = ['Race', 'Ethnicity', 'division_census', 'Sex', 'MOUDInitiationLocation', 'MOUDInitiationType']
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna('Unknown', inplace=True)

    # 2. Handle missing numerical values with median imputation
    numeric_median_cols = ['Total_Miles', 'Total_Minutes', 'CtDaysCoveredAntidepEpisode', 'CtDaysCoveredBenzoEpisode']
    for col in numeric_median_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # 3. Apply KNN imputation for selected numerical columns
    knn_columns = ['RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4', 'RPL_THEMES']
    knn_imputer = KNNImputer(n_neighbors=5)
    
    # Ensure that all columns exist before applying KNN
    knn_columns = [col for col in knn_columns if col in df.columns]
    if knn_columns:
        df[knn_columns] = knn_imputer.fit_transform(df[knn_columns])

    # 4. Fill missing 'CountOfOutpatVisit' using the mean where YearOfInitiation is between 2018 and 2024
    if 'CountOfOutpatVisit' in df.columns and 'YearOfInitiation' in df.columns:
        mask = (df['YearOfInitiation'] >= 2018) & (df['YearOfInitiation'] <= 2024)
        mean_value = df.loc[mask, 'CountOfOutpatVisit'].mean()
        df.loc[mask, 'CountOfOutpatVisit'] = df.loc[mask, 'CountOfOutpatVisit'].fillna(mean_value)

    return df



def standardize_columns(df):
    """
    Standardizes values in the 'Race' and 'Ethnicity' columns using predefined mappings.
    Additionally, it converts all races except 'White' and 'Black or African American' to 'Other Race'.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with standardized 'Race' and 'Ethnicity' columns.
    """

    # Mapping for Race column
    race_mapping = {
        'Unknown by Patient': 'Unknown',
        'Missing': 'Unknown',
        'Declined to Answer': 'Unknown',
        'UNKNOWN BY PATIENT': 'Unknown',
        'DECLINED TO ANSWER': 'Unknown',
        'Black or African American': 'Black or African American',
        'BLACK OR AFRICAN AMERICAN': 'Black or African American',
        'Black': 'Black or African American',
        'WHITE NOT OF HISP ORIG': 'White',
        'WHITE': 'White',
        'Multiracial': 'More than one Race',
        'ASIAN': 'Asian',
        'Multiple Race Selected': 'More than one Race',
        'Alaskan Native or American Indian': 'American Indian or Alaska Native',
        'AMERICAN INDIAN OR ALASKA NATIVE': 'American Indian or Alaska Native',
        'American Indian or Alaska Native': 'American Indian or Alaska Native',
        'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Native Hawaiian or Pacific Islander',
        'Native Hawaiian or Other Pacific Islande': 'Native Hawaiian or Pacific Islander',
    }

    # Mapping for Ethnicity column
    ethnicity_mapping = {
        'Unknown by Patient': 'Unknown',
        'UNKNOWN BY PATIENT': 'Unknown',
        'Missing': 'Unknown',
        'Declined to Answer': 'Unknown',
        'Declined to answer': 'Unknown',
        'DECLINED TO ANSWER': 'Unknown',
        'NOT HISPANIC OR LATINO': 'Not Hispanic or Latino',
        'Not Hispanic or Latino': 'Not Hispanic or Latino',
        'Not Hispanic or  Latino': 'Not Hispanic or Latino',
        'Not Hispanic': 'Not Hispanic or Latino',
        'HISPANIC OR LATINO': 'Hispanic or Latino',
        'Hispanic or Latino': 'Hispanic or Latino',
        'Hispanic': 'Hispanic or Latino',
    }

    # Standardize Race column
    if 'Race' in df.columns:
        df['Race'] = df['Race'].replace(race_mapping)
        df['Race'] = df['Race'].apply(lambda x: x if x in ['White', 'Black or African American'] else 'Other Race')

    # Standardize Ethnicity column
    if 'Ethnicity' in df.columns:
        df['Ethnicity'] = df['Ethnicity'].replace(ethnicity_mapping)

    return df


def merge_race_and_ethnicity(df):
    """
    Merges Race and Ethnicity into a single column in the given DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing race and ethnicity columns.

    Returns:
    --------
    pd.DataFrame
        The updated DataFrame with an additional merged column.
    """

    def _merge_logic(row):
        race = row['Race']
        ethnicity = row['Ethnicity']

        if ethnicity == 'Hispanic or Latino':
            return 'Hispanic or Latino'
        elif race == 'Unknown' and ethnicity == 'Unknown':
            return 'Unknown'
        elif race == 'Unknown' and ethnicity == 'Hispanic or Latino':
            return ethnicity
        elif race == 'Unknown' and ethnicity == 'Not Hispanic or Latino':
            return race
        elif ethnicity == 'Unknown':
            return race
        else:
            return str(race)

    df['race_ethnicity'] = df.apply(_merge_logic, axis=1)
    return df


def create_failure_rates_inplace(df):
    """
    Creates failure-rate columns (Bup, Meth, Nalt) in the given DataFrame by
    dividing 'Failed_*_Episodes' by 'Previous_*_Episodes'. Division by zero is 
    handled by replacing the denominator with NaN, then filling those NaNs with 0.

    This function modifies the input DataFrame in place.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.

    Returns:
    --------
    None
    """

    df['Previous_Bup_Episodes'].replace(0, np.nan, inplace=True)
    df['Previous_Meth_Episodes'].replace(0, np.nan, inplace=True)
    df['Previous_Nalt_Episodes'].replace(0, np.nan, inplace=True)

    df['Failed_Bup_Rate'] = df['Failed_Bup_Episodes'] / df['Previous_Bup_Episodes']
    df['Failed_Meth_Rate'] = df['Failed_Meth_Episodes'] / df['Previous_Meth_Episodes']
    df['Failed_Nalt_Rate'] = df['Failed_Nalt_Episodes'] / df['Previous_Nalt_Episodes']

    df[['Failed_Bup_Rate', 'Failed_Meth_Rate', 'Failed_Nalt_Rate']].fillna(0, inplace=True)

    df.drop(columns=['Failed_Bup_Episodes', 'Failed_Meth_Episodes', 'Failed_Nalt_Episodes'], inplace=True)

def one_hot_encode(df):
    """
    Performs one-hot encoding on categorical columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with categorical columns one-hot encoded.
    """
    categorical_columns = ['Sex', 'MOUDInitiationLocation', 'MOUDInitiationType', 'race_ethnicity']
    return pd.get_dummies(df, columns=categorical_columns)

def drop_specific_columns(df):
    """
    Drops specified columns from the given DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame

    Returns:
    --------
    pd.DataFrame
        The DataFrame with the specified columns removed.
    """
    columns_to_drop = ['Id', 'interval', 'episode_number', 'TotalDaysCovered', 'division_census', 'Total_Minutes', 
                       'RPL_THEMES', 'Sex_Prefer not to answer', 'Sex_F','MOUDInitiationLocation_I','MOUDInitiationType_Bup', 
                       'race_ethnicity_Other']
    return df.drop(columns=columns_to_drop, errors='ignore')

def main(df):
    """
    Runs the complete data preprocessing pipeline:
    - Handles missing values
    - Standardizes race and ethnicity
    - Merges race and ethnicity
    - Creates failure rates
    - Drops unnecessary columns
    - Applies one-hot encoding

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to be preprocessed.

    Returns:
    --------
    pd.DataFrame
        The fully preprocessed DataFrame.
    """

    try:
        # Handle missing values only if NaNs exist
        if df.isnull().sum().sum() > 0:
            df = handle_missing_values(df)
        else:
            logging.info("Skipping handle_missing_values: No missing values found.")

        # Standardize race and ethnicity only if both columns exist
        if {'Race', 'Ethnicity'}.issubset(df.columns):
            df = standardize_columns(df)
        else:
            logging.info("Skipping standardize_columns: Race/Ethnicity columns not found.")

        # Merge race and ethnicity only if 'race_ethnicity' column doesn't exist
        if 'race_ethnicity' not in df.columns:
            df = merge_race_and_ethnicity(df)
        else:
            logging.info("Skipping merge_race_and_ethnicity: race_ethnicity column already exists.")

        # Create failure rates only if 'Failed_*' columns exist
        failure_cols = {'Failed_Bup_Episodes', 'Failed_Meth_Episodes', 'Failed_Nalt_Episodes'}
        if failure_cols.intersection(df.columns):
            create_failure_rates_inplace(df)  # Modifies df in place
        else:
            logging.info("Skipping create_failure_rates_inplace: No Failed_* columns found.")

        # Drop specific columns only if they exist
        drop_cols = {'Id', 'interval', 'episode_number', 'TotalDaysCovered', 'division_census', 'Total_Minutes', 'RPL_THEMES'}
        if drop_cols.intersection(df.columns):
            df = drop_specific_columns(df)
        else:
            logging.info("Skipping drop_specific_columns: No columns to drop found.")

        # Perform one-hot encoding only if categorical columns exist
        categorical_columns = {'Sex', 'MOUDInitiationLocation', 'MOUDInitiationType', 'race_ethnicity'}
        if categorical_columns.intersection(df.columns):
            df = one_hot_encode(df)
        else:
            logging.info("Skipping one_hot_encode: No categorical columns found.")

        logging.info("✅ Data preprocessing completed successfully!")
        return df  # Return the final processed DataFrame

    except Exception as e:
        logging.error(f"❌ Error during data preprocessing: {str(e)}")
        return df  

# ---------------------------------------
# Example Usage: Run the script
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('complete_final_dataset.csv', index_col =False)

    # Run the preprocessing pipeline
    df = main(df)

    # Save the cleaned dataset
    df.to_csv("processed_data.csv", index=False)
    logging.info("✅ Processed dataset saved as 'processed_data.csv'.")
