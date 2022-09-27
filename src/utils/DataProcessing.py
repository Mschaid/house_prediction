
import numpy as np
import pandas as pd
import re


def tweak_homes(df):

    def null_counts(df: pd.DataFrame, upper_lim: float, lower_lim=0) -> tuple:
        """Calculate the percentage of missing values in each column and return:
                a list of column names that should be imputed. 
                a list of column names that should be dropped.
        """
    # calculate missing values percentage and return
        null_counts = pd.DataFrame(
            df.isnull().sum()/df.shape[0], columns=['null_counts'])
        cols_impute = null_counts.query(
            "null_counts > @lower_lim & null_counts < @upper_lim").index.to_list()
        cols_drop = null_counts.query(
            "null_counts >= @upper_lim").index.to_list()
        return cols_impute, cols_drop
    # null_cols = null_counts(df, 0.05)

    def to_category(df_) -> list:
        return df_.astype({c: 'category' for c in df_.select_dtypes('object').columns})

    def group_col_names(df, term):
        return [col for col in df.columns if re.search(term, col)]

    return (df
            # drop columns with more than 5% missing values
            .drop(columns=null_counts(df, 0.05)[1])
            # drop identifier columns that are not useful for modeling
            .drop(columns=['PID', 'Order'])
            # impute numerical columns and fill with mean value
            .fillna(value={c: df[c].mean() for c in df.select_dtypes('number').columns})
            # rename columns to lowercase and replace spaces with underscores
            .rename(lambda col: col.replace(' ', '_').lower(), axis=1)
            # convert object columns to category
            .pipe(lambda df_: to_category(df_))
            .assign(year_until_remod=lambda df_: df_['year_remod/add'] - df_['year_built'],  # years between remod and built
                    total_sf=lambda df_: df_[group_col_names(df_, 'sf')].sum(
                        axis=1),  # total square footage
                    total_area=lambda df_: df_[group_col_names(
                        df_, 'area')].sum(axis=1),  # total area
                    )
            )

def get_dummies(df):
       cat_to_drop = df.select_dtypes('category').columns.tolist()

       neigh_dummies = (pd.get_dummies(df['neighborhood'])
                     .rename(lambda col: col.lower(), axis=1)
                     )      
       feature_dummies = pd.concat([df, neigh_dummies], axis=1).drop(columns=cat_to_drop)
       return pd.concat([df.select_dtypes('number'), feature_dummies])
   
