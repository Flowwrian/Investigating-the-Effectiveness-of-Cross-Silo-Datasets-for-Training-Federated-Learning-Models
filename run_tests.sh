
#!/bin/sh
#change into scripts directory
cd scripts/

#define arguments
declare -a datasets=("covid" "weather")
declare -a clients=(2 4)
declare -a rounds=(2 4)
declare -a sk_models=("linear regression", "linearSVR")
declare -a tf_models=("MLP", "LSTM", "CNN")

declare -a stations=("berlin_alexanderplatz" "frankfurt_am_main_westend" "hamburg_airport" "leipzig" "muenchen" "potsdam")
declare -a covid_features=("new_cases", "weekly_hosp_admissions", "new_deaths", "weekly_icu_admissions")

declare -a epochs=(2 10)
declare -a hidden_layers=(2 3 4)
declare -a weather_batch_size=(16 32)
declare -a covid_batch_size=(500 1000)


#weather dataset
for c in "${clients[@]}"
do
    for r in "${rounds[@]}"
    do
        #sklearn models
        for m in "${sk_models[@]}"
        do
            python main.py --m "$m" --d weather --attributes temp rhum dwpt --r "$r" --c "$c" --sa 10000 --l MAE --batch_size 32 --standardize True --hidden_layers 5 --stations "${stations[@]:0:$c}" --log True
        done

        #tf models
        for m in "${tf_models[@]}"
        do
            for e in "${epochs[@]}"
            do
                for h in "${hidden_layers[@]}"
                do
                    for b in "${weather_batch_size[@]}"
                    do
                        python main.py --m "$m" --d weather --attributes temp rhum dwpt --r "$r" --c "$c" --sa 10000 --l MAE --batch_size 32 --standardize True --stations "${stations[@]:0:$c}" --epochs "$e" --hidden_layers "$h" --batch_size "$b" --log True
                    done
                done
            done
        done
    done
done

#covid dataset
for c in "${clients[@]}"
do
    for e in "${epochs[@]}"
    do
        for m in "${tf_models[@]}"
        do
            for h in "${hidden_layers[@]}"
            do
                for b in "${covid_batch_size[@]}"
                do
                    python main.py --m "$m" --d covid --attributes "${covid_features[@]:0:$c}" --r "$e" --c "$c" --sa 100000 --l MAE --standardize True --hidden_layers "$h" --batch_size "$b" --log True
                done
            done
        done
    done
done

cd ..