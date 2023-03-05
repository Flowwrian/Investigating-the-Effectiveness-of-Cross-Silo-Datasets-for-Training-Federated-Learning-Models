
#!/bin/sh
#change into scripts directory
cd scripts/

#define arguments
declare -a tf_models=("MLP" "LSTM" "CNN")
declare -a covid_features=("new_cases" "weekly_hosp_admissions" "new_deaths" "weekly_icu_admissions")
declare -a vfl_clients=(2 3 4)

#best MLP; two features
for i in 1 2 3 4
do
    python main.py --m MLP --d covid --attributes "new_cases" "weekly_hosp_admissions" --r 10 --c 2 --sa 100000 --l MAE --batch_size 1000 --standardize True --hidden_layers 2 --log True
done

#best LSTM; two features
for i in 1 2 3 4
do
    python main.py --m LSTM --d covid --attributes "new_cases" "weekly_hosp_admissions" --r 10 --c 2 --sa 100000 --l MAE --batch_size 1000 --standardize True --hidden_layers 3 --log True
done

#best CNN; two features
for i in 1 2 3 4
do
    python main.py --m CNN --d covid --attributes "new_cases" "weekly_hosp_admissions" --r 10 --c 2 --sa 100000 --l MAE --batch_size 500 --standardize True --hidden_layers 3 --log True
done




#best MLP; three features
for i in 1 2 3 4
do
    python main.py --m MLP --d covid --attributes "new_cases" "weekly_hosp_admissions" "new_deaths" --r 10 --c 3 --sa 100000 --l MAE --batch_size 1000 --standardize True --hidden_layers 2 --log True
done

#best LSTM; three features
for i in 1 2 3 4
do
    python main.py --m LSTM --d covid --attributes "new_cases" "weekly_hosp_admissions" "new_deaths" --r 10 --c 3 --sa 100000 --l MAE --batch_size 1000 --standardize True --hidden_layers 2 --log True
done

#best CNN; three features
for i in 1 2 3 4
do
    python main.py --m CNN --d covid --attributes "new_cases" "weekly_hosp_admissions" "new_deaths" --r 10 --c 3 --sa 100000 --l MAE --batch_size 1000 --standardize True --hidden_layers 2 --log True
done




#best MLP; four features
for i in 1 2 3 4
do
    python main.py --m MLP --d covid --attributes "new_cases" "weekly_hosp_admissions" "new_deaths" "weekly_icu_admissions" --r 10 --c 4 --sa 100000 --l MAE --batch_size 1000 --standardize True --hidden_layers 2 --log True
done

#best LSTM; four features
for i in 1 2 3 4
do
    python main.py --m LSTM --d covid --attributes "new_cases" "weekly_hosp_admissions" "new_deaths" "weekly_icu_admissions" --r 10 --c 4 --sa 100000 --l MAE --batch_size 1000 --standardize True --hidden_layers 2 --log True
done

#best CNN; four features
for i in 1 2 3 4
do
    python main.py --m CNN --d covid --attributes "new_cases" "weekly_hosp_admissions" "new_deaths" "weekly_icu_admissions" --r 10 --c 4 --sa 100000 --l MAE --batch_size 500 --standardize True --hidden_layers 2 --log True
done

cd ..