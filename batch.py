"""
This file writes a limited batch of the full dataset to a new csv. In order to run, you must have the decompressed version on your machine.

Run example:
    python3 batch.py
"""

import os

import pandas as pd

DATA_PATH = '../'
FILE_NAME = 'TWOSIDES.csv'
n = 10  # every 10th line = 10% of the lines
df = pd.read_csv(
    os.path.join(DATA_PATH, FILE_NAME),
    header=0, skiprows=lambda i: i % n != 0,
    usecols=['drug_1_concept_name', 'drug_2_concept_name',
    'condition_concept_name', 'PRR', 'PRR_error'],
    low_memory=False)

# Fill null values with zeros
df['PRR'] = pd.to_numeric(df['PRR'], errors='coerce')
df['PRR_error'] = pd.to_numeric(df['PRR'], errors='coerce')
df = df.fillna(0)

# Concatenate the drug names to form a single entitiy, convert to lowercase
df['joint_drug_name'] = df['drug_1_concept_name'] + "_" + df['drug_2_concept_name']
df['joint_drug_name'] = df['joint_drug_name'].str.lower()
df['condition_concept_name'] = df['condition_concept_name'].str.lower()

# Write to new file
df[['joint_drug_name', 'condition_concept_name', 'PRR', 'PRR_error']].to_csv('twosides-10pct.csv')


