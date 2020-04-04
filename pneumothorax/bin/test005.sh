model=model005
gpu=1

conf=./conf/${model}.py

python -m src.main test ${conf} --gpu ${gpu}
