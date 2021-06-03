import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os.path
from os import path
from datasets import GetDataset
import random
import torch

import sys
sys.path.insert(0, '..')
import arc

def assess_predictions(S, X, y, kept_classes):
    # Marginal coverage
    coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    # Average length
    length = np.mean([len(S[i]) for i in range(len(y))])
    # Average length conditional on coverage
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    length_cover = np.mean([len(S[i]) for i in idx_cover])
    # Conditional coverage (WSC)
    # cond_coverage = arc.coverage.wsc_unbiased(X, y, S)
    cond_coverage = 1.0

    # Class Conditional Coverage
    labels = np.sort(np.unique(y))
    if kept_classes is not None:
        labels = labels[kept_classes]
    cc_coverage = np.zeros(len(labels))
    cc_size = np.zeros(len(labels))
    for i, label in enumerate(labels):
        idxs = np.argwhere(y == label)[:, 0]
        S_filtered = [S[idx] for idx in idxs]
        X_filtered = X[idxs]
        y_filtered = y[idxs]

        cc_coverage[i] = np.mean([y_filtered[i] in S_filtered[i] for i in range(len(y_filtered))])
        cc_size[i] = np.mean([len(S_filtered[i]) for i in range(len(y_filtered))])

    # Combine results
    out = pd.DataFrame({'Marginal Coverage': [coverage], 'Minimum CC Coverage': [min(cc_coverage)], 'Average CC Set Size': [length], 'Gamma': (max(y)+1) - len(labels)})
    return out

def collect_predictions(S, X, y, condition_on):
    cover = np.array([y[i] in S[i] for i in range(len(y))])
    length = np.array([len(S[i]) for i in range(len(y))])
    out = pd.DataFrame({'Cover': cover, 'Length': length})
    var_name = "y"
    out[var_name] = y
    return out

def run_experiment(out_dir, dataset_name, dataset_base_path, n_train, alpha, experiment, dataset_cache=None, gamma=0):

    # load dataset
    if dataset_cache is None:
        X, y = GetDataset(dataset_name, dataset_base_path)
        y = y.astype(np.long)
    else:
        X, y = dataset_cache
        y = y.astype(np.long)

    # Determine output file
    out_file_1 = out_dir + "/summary.csv"
    out_file_2 = out_dir + "/full.csv"
    out_files = [out_file_1, out_file_2]
    print(out_files)
    
    # Random state for this experiment
    random_state = 2020 + experiment
    
    # Set random seed
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    
    # List of calibration methods to be compared
    methods = {
        'APS': arc.methods.SplitConformal,
        'CCAPS': arc.methods.SplitConformalCC,
        # 'CCAPSReborn': arc.methods.SplitConformalCCReborn
     #    'CV+': arc.methods.CVPlus,
     #    'JK+': arc.methods.JackknifePlus,
     #    'HCC': arc.others.SplitConformalHomogeneous,
     #    'CQC': arc.others.CQC,
     #    'CQCRF': arc.others.CQCRF
    } 

    # List of black boxes to be compared
    if dataset_name != "imagenet":
        black_boxes = {
                       # 'SVC': arc.black_boxes.SVC(random_state=random_state),
                       'RFC': arc.black_boxes.RFC(n_estimators=100,
                                                  criterion="gini",
                                                  max_depth=None,
                                                  max_features="auto",
                                                  min_samples_leaf=3,
                                                  random_state=random_state),
                       'NNet': arc.black_boxes.NNet(hidden_layer_sizes=64,
                                                    batch_size=128,
                                                    learning_rate_init=0.01,
                                                    max_iter=20,
                                                    random_state=random_state)
                      }
    else:
        black_boxes = {"ResNet152": arc.black_boxes.Identity()}

    # Which special variables should we condition on to compute conditional coverage?
    condition_on = [0]
    
    # Total number of samples
    n_test = X.shape[0] - n_train

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=random_state)
    
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]

    # Load pre-computed results
    if path.exists(out_files[0]) & path.exists(out_files[1]):
        results = pd.read_csv(out_files[0])
        results_full = pd.read_csv(out_files[1])
    else:
        results = pd.DataFrame()
        results_full = pd.DataFrame()
        
    for box_name in black_boxes:
        black_box = black_boxes[box_name]
        for method_name in methods:
            # Skip if this experiment has already been run
            if results.shape[0] > 0:
                found  = (results['Method']==method_name)
                found &= (results['Black box']==box_name)
                found &= (results['Experiment']==experiment)
                found &= (results['Nominal']==1-alpha)
                found &= (results['n_train']==n_train)
                found &= (results['n_test']==n_test)
                found &= (results['dataset']==dataset_name)
            else:
                found = 0
            if np.sum(found) > 0:
                print("Skipping experiment with black-box {} and method {}...".format(box_name, method_name))
                sys.stdout.flush()
                continue

            print("Running experiment with black-box {} and method {}...".format(box_name, method_name))
            sys.stdout.flush()

            # Train classification method
            method = methods[method_name](X_train, y_train, black_box, alpha, random_state=random_state,
                                          verbose=True)
            # Apply classification method
            S, kept_classes = method.predict(X_test, gamma=gamma)

            # Evaluate results
            res = assess_predictions(S, X_test, y_test, kept_classes)
            # Add information about this experiment
            res['Method'] = method_name
            res['Black box'] = box_name
            res['Experiment'] = experiment
            res['Nominal'] = 1-alpha
            res['n_train'] = n_train
            res['n_test'] = n_test
            res['dataset'] = dataset_name

            # Evaluate results (conditional)
            res_full = collect_predictions(S, X_test, y_test, condition_on)
            # Add information about this experiment
            res_full['Method'] = method_name
            res_full['Black box'] = box_name
            res_full['Experiment'] = experiment
            res_full['Nominal'] = 1-alpha
            res_full['n_train'] = n_train
            res_full['n_test'] = n_test
            res_full['dataset'] = dataset_name

            # Add results to the list
            results = results.append(res)
            results_full = results_full.append(res_full)

            # Write results on output files
            if len(out_files) == 2:
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
            
                results.to_csv(out_files[0], index=False, float_format="%.4f")
                print("Updated summary of results on\n {}".format(out_files[0]))
                results_full.to_csv(out_files[1], index=False, float_format="%.4f")
                print("Updated full results on\n {}".format(out_files[1]))
                sys.stdout.flush()

    return results, results_full
