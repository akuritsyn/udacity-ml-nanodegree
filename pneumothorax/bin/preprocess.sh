mkdir -p cache  # model data/submission

python -m src.preprocess.make_folds --input ./input/stage_2_train.csv --output ./cache/train_folds.pkl --n_folds 5 --seed 69