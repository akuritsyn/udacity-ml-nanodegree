model=model004
gpu=0
fold=1
conf=./conf/${model}.py
#--debug 100

python -m src.main train ${conf} --fold ${fold} --gpu ${gpu}
