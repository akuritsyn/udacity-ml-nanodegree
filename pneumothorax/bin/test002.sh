model=model002
gpu=0
#fold=1
conf=./conf/${model}.py

python -m src.main test ${conf} --gpu ${gpu}
