# Scalable Recommendation Systems for Modeling Drug Interactions with Matrix Factorization

## About
This resposity contains the code used to create a recommendation system for predicting side effect propensity in pharmacological combinations. Several matrix factorization methods are trained and tested on Twosides data, parameter-optimized using GridsearchCV, and then compared against each other for efficiency and performance.

## Requirements
Certain packages require Python 3.6 or higher. To install requirements, run:
```
virtualenv env
source env/bin/activate
(env) > pip3 install -r requirements.txt
```

## Running the Code
The main script is contained in `final.py`. To run on the default mini-batch of data, run the following command in the terminal:
```
python3 final.py
```

## Output
Output is currently limited to logging results to the console. Future work includes formatting inter/intra model tables to report performance and process time. There is some work started in `plotting.py` to visualize the results of the gridsearch object.
