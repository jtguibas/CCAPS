

from run_experiment_real_data import run_experiment
from datasets import GetDataset

alpha = 0.1

DATASET_LIST   = ["imagenet"]
N_TRAIN_LIST = [25000]

# Data sets directory 
dataset_base_path = "/future/u/alexder/other/cs229/mydata/classification_data/"

# Where to write results
out_dir = "./results-imagenet"
dataset_cache = GetDataset("imagenet", dataset_base_path)

for EXP_id in range(50):
    print(f"{EXP_id}/50")
    for N_TRAIN_LIST_id in range(len(N_TRAIN_LIST)):
       for DATASET_LIST_id in range(len(DATASET_LIST)):
 
           dataset_name = DATASET_LIST[DATASET_LIST_id]
           n_train = N_TRAIN_LIST[N_TRAIN_LIST_id]

           
           run_experiment(out_dir,
                          dataset_name,
                          dataset_base_path,
                          n_train,
                          alpha=alpha,
                          experiment=EXP_id)
