"""
This file accompanies the Maxtrix Factorization paper for Drug Interaaction analysis. It is the main file for running an inter/intra model comarision of MF techniques on drug-drug-reaction data.

Code and ideas generously appropriated from examples at the the following:
https://github.com/NicolasHug/Surprise

Author: Tim Hannifan
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import numpy as np
import pandas as pd

# pip3 install scikit-surprise; see requirements.txt
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

from collections import defaultdict
from tabulate import tabulate
import matplotlib.pyplot as plt


# Import data result from batch.py. The default is a .001% of total data available in the full dataset.
data_path = './'
filename = 'twosides-mini.csv'
df = pd.read_csv(
    os.path.join(data_path, filename),
    usecols=['joint_drug_name', 'condition_concept_name', 'PRR'],
    low_memory=False)

# Create reader object with correct scale
reader = Reader(rating_scale=(min(df["PRR"]), max(df["PRR"])))
# The columns correspond to drug_names, reaction and PRR (in that order).
data = Dataset.load_from_df(df[['joint_drug_name', 'condition_concept_name', 'PRR']], reader)

# Define param grid for each algorithm
pg_svd = {'n_factors': [10, 20, 50], 'n_epochs': [10, 20],
            'lr_all': [0.002, 0.005], 'reg_all': [0.2, 0.4]}
pg_svd_pp = {'n_factors': [10, 20, 50], 'n_epochs': [10, 20],
                'lr_all': [0.002, 0.005], 'reg_all': [0.2, 0.4]}
pg_svd_nmf = {'n_factors': [10, 20, 50], 'n_epochs': [10, 20],
                'biased': [True, False]}

for model_grid in [(BaselineOnly, {}),(SVD, pg_svd), (SVDpp, pg_svd_pp), (NMF, pg_svd_nmf)]:

    model, param_grid = model_grid
    print('\nStarting new model--------------------', model)
    gs = GridSearchCV(model, param_grid, measures=['rmse'], cv=5)
    gs.fit(data)

    # Best score `
    print('Lowest RMS achieved: {}'.format(gs.best_score['rmse']))
    # combination of parameters that gave the best RMSE score
    print('Best parameters combination: {}'.format(gs.best_params['rmse']))


# TODO: implement table output for Latex using gridsearch results

# colstokeep = ['mean_test_rmse','mean_test_rmse']
# table = [[] for _ in range(len(colstokeep))]
# for i in range(len(colstokeep)):
#     for key in colstokeep:
#         # print(gs.cv_results[key][i])
#         table[i].append(gs.cv_results[key][i])

# header = gs.cv_results.keys()
# print(header)
# plot_grid_search(gs.cv_results, gs.cv_results['n_factors'], gs.cv_results['n_epochs'], 'n_factors',
#     'n_epochs')
#     # 'lr_all', 'reg_all',
#     # 'biased')


# print(tabulate(table, colstokeep, tablefmt="latex"))



