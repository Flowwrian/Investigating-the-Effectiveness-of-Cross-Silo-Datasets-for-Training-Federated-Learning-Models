from enum import Enum
from datetime import datetime

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import flwr


class Dataset(Enum):
    Other = 0
    Covid = 1
    Weather = 2

def get_samples(data: DataFrame, n: int, dataset: Dataset = Dataset.Other, max_samples: int = 100000) -> pd.Series:
    """
    Generate samples from a given dataset. The samples are created by using a rolling window.
    Args:
        data (DataFrame): The dataset sampling from.
        n (int): Number of records per sample.
        dataset (Dataset): Enum for specifing which dataset you want to sample from.
            Dataset.Covid: owid-covid-data.csv
            Dataset.Weather: Weather data from ___
            Dataset.Other: other datasets
        max_samples (int): Maximum amount returned samples.
    """

    samples = list()
    num_of_rows = len(data.index)

    if dataset == Dataset.Covid: #Ensure that only data from the same country gets into one sample
        for i in range(num_of_rows):
            if i+n > num_of_rows or len(samples) == max_samples:
                break
            if _check_covid_dataset(data.iloc[range(i,i+n)]):
                samples.append(data.iloc[range(i,i+n)])
    #TODO write logic for weather/ other datasets

    return pd.Series(samples)

def _check_covid_dataset(data: DataFrame):
    return (data.location == data.location.iloc[0]).all()


def sample_split(data: pd.Series, testing_percentage: float, x_attributes: list[str], y_attribute: str):
    """
    Splits sample data in x_train, x_test, y_train and y_test. Only suitable for time series data.
    The function takes the last entry in each sample as y value.

    Args:
        data (pd.Series): Sample data.
        testing_percentage (float): The percentage of data used for testing. Must be between 0 and 1.
        x_attributes (list[str]): The exogene variables used for input.
        y_attribute (str): The endogene variable (the expected model output).
    """


    sample_length = len(data.iloc[0].index)
    x_data = []
    y_data = []

    for i in range(len(data)):
        current_sample = data.iloc[i]
        x_data.append(current_sample.iloc[range(0, sample_length - 1)][x_attributes])
        y_data.append(current_sample.iloc[sample_length - 1][y_attribute])

    #Scikit-Learn function used for convinience
    return train_test_split(x_data, y_data, test_size=testing_percentage, random_state=0)


#used for testing purposes
if __name__ == "__main__":
    startTime = datetime.now()
    test_data = pd.read_csv("datasets\\horizontal\\covid\\owid-covid-data.csv")
    result = get_samples(test_data, 3, Dataset.Covid)

    print(len(result))
    print(datetime.now() - startTime)