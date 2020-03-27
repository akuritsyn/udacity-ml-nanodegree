model=model001
gpu=1
fold=1
conf=./conf/${model}.py

python -m src.main train ${conf} --fold ${fold} --gpu ${gpu}
