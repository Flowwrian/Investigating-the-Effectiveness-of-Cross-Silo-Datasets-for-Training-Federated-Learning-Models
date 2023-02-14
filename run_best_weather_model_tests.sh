
#!/bin/sh
#change into scripts directory
cd scripts/

#define arguments
declare -a sk_models=("linear regression" "linearSVR")
declare -a tf_models=("MLP" "LSTM" "CNN")

declare -a stations=("berlin_alexanderplatz" "frankfurt_am_main_westend" "hamburg_airport" "leipzig" "muenchen" "potsdam" "hannover" "koeln_bonn_airport" "stuttgart_schnarrenberg" "weimar")



#best sklearn models
for m in "${sk_models[@]}"
do
    for i in 1 2 3 4
    do
        python main.py --m "$m" --d weather --attributes temp rhum dwpt --r 2 --c 3 --sa 10000 --l MAE --batch_size 32 --standardize True --hidden_layers 5 --stations ${stations[@]:0:3} --log True
    done
done

#best MLP
for i in 1 2 3 4
do
    python main.py --m MLP --d weather --attributes temp rhum dwpt --r 4 --c 3 --sa 10000 --l MAE --batch_size 16 --standardize True --hidden_layers 4 --stations ${stations[@]:0:3} --log True
done

#best LSTM
for i in 1 2 3 4
do
    python main.py --m LSTM --d weather --attributes temp rhum dwpt --r 2 --c 3 --sa 10000 --l MAE --batch_size 16 --standardize True --hidden_layers 2 --stations ${stations[@]:0:3} --log True
done

#best CNN
for i in 1 2 3 4
do
    python main.py --m CNN --d weather --attributes temp rhum dwpt --r 2 --c 2 --sa 10000 --l MAE --batch_size 16 --standardize True --hidden_layers 3 --stations ${stations[@]:0:2} --log True
done
