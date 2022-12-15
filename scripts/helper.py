from enum import Enum
from datetime import datetime

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


class Library(Enum):
    Tensorflow = 1
    Sklearn = 2


class Dataset(Enum):
    Other = 0
    Covid = 1
    Weather = 2

def get_samples(data: DataFrame, n: int, testing_percentage: float, x_attributes: list[str], y_attribute: str, dataset: Dataset = Dataset.Other, max_samples: int = 100000):
    """
    Generate samples from a given dataset. The samples are created by using a rolling window.
    Args:
        data (DataFrame): The dataset sampling from.
        n (int): Number of records per sample.
        testing_percentage (float): The percentage of data used for testing. Must be between 0 and 1.
        x_attributes (list[str]): The exogene variables used for input.
        y_attribute (str): The endogene variable (the expected model output).
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
            
            #check for nans
            if y_attribute in x_attributes:
                new_data = data.iloc[range(i,i+n)][x_attributes]
            else:
                new_data = data.iloc[range(i,i+n)][[*x_attributes, y_attribute]]

            if _check_covid_dataset(data.iloc[range(i,i+n)]) and _check_for_nans(new_data):
                samples.append(new_data)
    #TODO write logic for weather/ other datasets

    data_series = pd.Series(samples)

    #logic from sample_split starts here
    sample_length = len(data_series.iloc[0].index)
    x_data = []
    y_data = []

    for i in range(len(data_series)):
        current_sample = data_series.iloc[i]
        new_x_data = current_sample.iloc[range(0, sample_length - 1)][x_attributes]
        new_y_data = current_sample.iloc[sample_length - 1][y_attribute]
        
        #if _check_for_nans(new_x_data) and _check_for_nans(new_y_data):
        x_data.append(new_x_data.to_numpy().flatten())
        y_data.append(new_y_data)

    #Scikit-Learn function used for convinience
    return train_test_split(x_data, y_data, test_size=testing_percentage, random_state=0)


def _check_covid_dataset(data: DataFrame) -> bool:
    return (data.location == data.location.iloc[0]).all()


def _check_for_nans(data: DataFrame) -> bool:
    return not data.isnull().values.any()


#used for testing purposes
if __name__ == "__main__":
    startTime = datetime.now()
    test_data = pd.read_csv("datasets\\horizontal\\covid\\owid-covid-data.csv")
    result = get_samples(test_data, 3, Dataset.Covid)

    print(len(result))
    print(datetime.now() - startTime)