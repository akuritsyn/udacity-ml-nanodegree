model=model007
gpu=0  
fold=0  # 0...4, change corresponding config too
conf=./conf/${model}.py
#--debug 100

python -m src.main train ${conf} --fold ${fold} --gpu ${gpu}
