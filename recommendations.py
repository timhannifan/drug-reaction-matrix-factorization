"""
This file is used to evaluate trained models against new instances and generate human-readable lists of predictions.

Code and ideas generously appropriated from examples at the the following:
https://github.com/NicolasHug/Surprise
"""
import os
import sys
import math
import pandas as pd
import numpy as np
from surprise import SVD
from surprise import NMF
from surprise import Dataset
from surprise import Reader
from helpers import get_top_n

# Import data result from batch.py. The default is a .001% of total data available in the full dataset.

DEFAULT_DATA_PATH = './data/tiny.csv'

def run(args):

    df = pd.read_csv(args['data_path'],
        usecols=['joint_drug_name', 'condition_concept_name', 'PRR',
            'PRR_error', 'mean_reporting_frequency'],
        dtype={
            'PRR': np.float,
            'PRR_error': np.float,
            'mean_reporting_frequency': np.float},
        low_memory=False)
    print('Full data shape: {}'.format(df.shape))


    # Filter our insignificant interactions
    df = df[( df['PRR'].apply(
        lambda x: math.log(x)) - 1.96 * df['PRR_error'] > math.log(2))]
    print('Significant data shape: {}'.format(df.shape))


    # Create reader object with correct scale
    reader = Reader(rating_scale=(min(df["mean_reporting_frequency"]), max(df["mean_reporting_frequency"])))
    # The columns correspond to drug_names, reaction and mean_reporting_frequency (in that order).
    data = Dataset.load_from_df(df[['joint_drug_name',
        'condition_concept_name', 'mean_reporting_frequency']], reader)

    # First train an SVD algorithm on the data
    trainset = data.build_full_trainset()
    algo = NMF(n_factors=3, n_epochs=5, biased=True)
    algo.fit(trainset)

    # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    top_n = get_top_n(predictions, n=5)

    # Print the recommended items for each user
    for uid, user_ratings in top_n.items():
        print('\n')
        print(uid, [iid for (iid, _) in user_ratings])

if __name__ == '__main__':
    args = {}
    if (len(sys.argv) == 2) and os.path.isfile(sys.argv[1]):
        args['data_path'] = sys.argv[1]
    else:
        args['data_path'] = DEFAULT_DATA_PATH

    run(args)
