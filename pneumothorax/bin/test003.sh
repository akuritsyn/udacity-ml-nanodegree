model=model003
gpu=1
#fold=1
conf=./conf/${model}.py

python -m src.main test ${conf} --gpu ${gpu}
