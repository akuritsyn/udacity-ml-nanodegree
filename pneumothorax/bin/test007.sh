model=model007
gpu=0

conf=./conf/${model}.py

python -m src.main test ${conf} --gpu ${gpu}
