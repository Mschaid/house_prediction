
import numpy as np
import pandas as pd
import re


class DataProcessing():
    """_summary_
    """

    def __init__(self, data_path, target):
        self.data_path = data_path
        self.target = target

    def load_data(self):
        """from data_path reads data"""
        self.data = pd.read_csv(self.data_path, delimiter='\t')
        return self

    def clean_data(self):
        df = self.data

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

        self.processed_data = (df
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
        return self

    def get_dummies(self, cols, df=None):
        if df is None:
            df = self.processed_data

        cat_to_drop = df.select_dtypes('category').columns.tolist()

        cols_dummies = pd.get_dummies(df[cols]).rename(
            lambda c: c.lower(), axis=1)

        self.processed_dummies = pd.concat(
            [df, cols_dummies], axis=1).drop(columns=cat_to_drop)

        return self

    def filter_features(self, df=None, lower_threshold=-0.1, upper_threshold=0.3):
        if df is None:
            df = self.processed_dummies

        corr = pd.DataFrame(df
                            .corr()['saleprice']
                            .sort_values(ascending=False)
                            )
        self.features = (corr
                         .query("saleprice < @lower_threshold or saleprice > @upper_threshold")
                         .drop('saleprice', axis=0)
                         .index
                         .to_list()
                         )
        self.filtered_data = self.processed_dummies[self.features]
        return self
