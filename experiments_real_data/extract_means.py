import numpy as np
import pandas as pd

def calc_mean(df, method, black_box, metric):
    df_copy = df[df['Method'] == method]
    df_copy = df_copy[df_copy['Black box'] == black_box]

    return df_copy[metric].mean()

def compile_results(path, black_box):
    df = pd.read_csv(path)
    for metric in ['Minimum CC Coverage', 'Average CC Set Size', 'Marginal Coverage']:
        for method in ['APS', 'CCAPS']:
            stat = calc_mean(df, method, black_box, metric)
            print(f'{metric}: {stat}, Method: {method}')

def synthetic_data(path):
    """
    Marginal Coverage,Minimum CC Coverage,Average CC Set Size,Method,Black box,Experiment,Nominal,n_train,n_test
    0.8997,0.8768,1.8708,APS,RFC,0,0.9000,25000,25000
    0.9144,0.9006,1.9887,CCAPS,RFC,0,0.9000,25000,25000
    """


