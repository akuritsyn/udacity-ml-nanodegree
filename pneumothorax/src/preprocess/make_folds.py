import argparse
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=13)
    return parser.parse_args()


def main():
    args = get_args()

    df = pd.read_csv(args.input)
    # df = df.drop_duplicates('ImageId')
    df['has_mask'] = np.where(df["EncodedPixels"] != "-1", 1, 0)

    kfold = StratifiedKFold(args.n_folds, shuffle=True, random_state=args.seed)

    folds_list = list(kfold.split(df["ImageId"], df["has_mask"]))

    with open(args.output, 'wb') as f:
        pickle.dump(folds_list, f)

    print('Saved folds to %s' % args.output)


if __name__ == '__main__':
    print(sys.argv)
    main()
