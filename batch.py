"""
This file writes a limited batch of the full dataset to a new csv. In order to run, you must have the decompressed version on your machine.

Example:
    python3 batch.py
"""

import os
import pandas as pd

DATA_PATH = '../'
INPUT_FNAME = 'TWOSIDES.csv'
OUTPUT_FNAME = 'twosides-lg.csv'

# Every 10th line = 10% of the lines
n = 10
df = pd.read_csv(
    os.path.join(DATA_PATH, INPUT_FNAME),
    header=0, skiprows=lambda i: i % n != 0,
    usecols=['drug_1_concept_name', 'drug_2_concept_name',
    'condition_concept_name', 'A', 'B', 'C', 'D',
    'PRR', 'PRR_error', 'mean_reporting_frequency'],
    low_memory=False)

# Convert numeric cols, fillna with zero
to_numeric = ['A', 'B', 'C', 'D', 'PRR', 'PRR_error',
    'mean_reporting_frequency']
for col in to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.fillna(0)

# Convert all string rows to lowercase
tolower = ['condition_concept_name', 'drug_1_concept_name',
    'drug_2_concept_name']
for col in tolower:
    df[col] = df[col].str.lower()

# Concatenate the drug names to form a single entitiy, convert to lowercase
df['joint_drug_name'] = (df['drug_1_concept_name'] + "_" +
    df['drug_2_concept_name'])

# Write to new file
df.to_csv(OUTPUT_FNAME, index=False)
