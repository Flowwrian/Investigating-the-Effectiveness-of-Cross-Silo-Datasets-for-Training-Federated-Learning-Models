from enum import Enum
from datetime import datetime

import pandas as pd
from pandas import DataFrame


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


#used for testing purposes
if __name__ == "__main__":
    startTime = datetime.now()
    test_data = pd.read_csv("datasets\\horizontal\\covid\\owid-covid-data.csv")
    result = get_samples(test_data, 3, Dataset.Covid)

    print(len(result))
    print(datetime.now() - startTime)