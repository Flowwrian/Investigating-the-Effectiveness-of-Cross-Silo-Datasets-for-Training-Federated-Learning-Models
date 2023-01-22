# Investigating the Effectiveness of Cross-silo Datasets for Training Federated Learning Models
This repository contains all the code used for the experiments presented in the thesis 'Investigating the Effectiveness of Cross-silo Datasets for Training Federated Learning Models'. The main logic is located in the `scripts` folder. The notebooks folder contains a number of Jupyter notebooks. `covid_dataset.ipynb` and `weather_dataset.ipynb` contain an analysis of the respective dataset, as well as the baseline models and the optimal hyperparameters for each model. `model_tests.ipynb` is used for internal testing purposes and is not relevant for users. `statsmodels.ipynb` was used to generate a graphic for the thesis and is not relevant as well. The `datasets` folder contains the covid and weather datasets used in the experiments.

## Usage
The code is written in Python 3.10.6.
To start a federated learning scenario execute the main.py located in the scripts folder (i.e. `python main.py`). The parameters of the experiment are defined at the top of the `main.py` file (for now). At the moment the parameters are changed by changing the string inside the variable.

## TODO
- Safe logs
- Implement Argsparse
- Implement Deep-learning architecture specifically designed for time-series regression
- Implement Decision Tree Regressor