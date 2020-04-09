model=model008
gpu=0

conf=./conf/${model}.py

python -m src.main test ${conf} --gpu ${gpu}
