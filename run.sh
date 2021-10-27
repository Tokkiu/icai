nohup python -u main.py --run_data "Beauty" --model "bert" --use_feature 1 > b-b-f.log &
nohup python -u main.py --run_data "Beauty" --model "bert" --use_feature 0 > b-b.log &
nohup python -u main.py --run_data "Beauty" --model "gru" --use_feature 1 > b-g-f.log &
nohup python -u main.py --run_data "Beauty" --model "gru" --use_feature 0 > b-g.log &