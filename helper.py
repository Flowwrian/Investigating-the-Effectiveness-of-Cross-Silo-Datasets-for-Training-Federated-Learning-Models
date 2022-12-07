from enum import Enum
from datetime import datetime

import pandas as pd
from pandas import DataFrame


class Dataset(Enum):
    Other = 0
    Covid = 1
    Weather = 2

def get_samples(data: DataFrame, n: int, dataset: Dataset = Dataset.Other, max_samples: int = 100000) -> list:
    samples = list()
    num_of_rows = len(data.index)

    if dataset == Dataset.Covid: #Ensure that only data from the same country gets into one sample
        for i in range(num_of_rows):
            if i+n > num_of_rows or len(samples) == max_samples:
                break
            if _check_covid_dataset(data.iloc[range(i,i+n)]):
                samples.append(data.iloc[range(i,i+n)])

    return samples

def _check_covid_dataset(data: DataFrame):
    return (data.location == data.location.iloc[0]).all()


#used for testing purposes
if __name__ == "__main__":
    startTime = datetime.now()
    test_data = pd.read_csv("datasets\\horizontal\\covid\\owid-covid-data.csv")
    result = get_samples(test_data, 3, Dataset.Covid)

    print(len(result))
    print(datetime.now() - startTime)