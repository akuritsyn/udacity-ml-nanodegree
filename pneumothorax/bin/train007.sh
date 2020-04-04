model=model007
gpu=1
fold=0
conf=./conf/${model}.py
#--debug 100

python -m src.main train ${conf} --fold ${fold} --gpu ${gpu}
