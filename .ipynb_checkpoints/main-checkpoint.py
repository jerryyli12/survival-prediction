from __future__ import print_function

import argparse
import pdb
import os
import math
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd

### Internal Imports
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler


def main(args):
	#### Create Results Directory
	if not os.path.isdir(args.results_dir):
		os.mkdir(args.results_dir)

	if args.k_start == -1:
		start = 0
	else:
		start = args.k_start
	if args.k_end == -1:
		end = args.k
	else:
		end = args.k_end

	latest_val_cindex = []
	folds = np.arange(start, end)

	### Start 5-Fold CV Evaluation.
	for i in folds:
		start = timer()
		seed_torch(args.seed)
		results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))
		if os.path.isfile(results_pkl_path):
			print("Skipping Split %d" % i)
			continue

		### Gets the Train + Val Dataset Loader.
		train_dataset, val_dataset = dataset.return_splits(from_id=False, 
				csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
		
		print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
		datasets = (train_dataset, val_dataset)
		
		args.omic_input_dim = 0

		### Run Train-Val on Survival Task.
		if args.task_type == 'survival':
			val_latest, cindex_latest = train(datasets, i, args)
			latest_val_cindex.append(cindex_latest)

		### Write Results for Each Split to PKL
		save_pkl(results_pkl_path, val_latest)
		end = timer()
		print('Fold %d Time: %f seconds' % (i, end - start))

	### Finish 5-Fold CV Evaluation.
	if args.task_type == 'survival':
		print(latest_val_cindex)
		results_latest_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex})

	if len(folds) != args.k:
		save_name = 'summary_partial_{}_{}.csv'.format(start, end)
	else:
		save_name = 'summary.csv'

	results_latest_df.to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))

### Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir',   type=str, default='path/to/data_root_dir', help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', 			     type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir',     type=str, default='./results', help='Results directory (Default: ./results)')
parser.add_argument('--which_splits',    type=str, default='5foldcv', help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir',       type=str, default='tcga_brca', help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite',     	 action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')

### Model Parameters.
parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn', 'pyramid', 'mutation', 'rnaseq', 'multimodal'], default='coattn', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion',          type=str, choices=['None', 'concat', 'bilinear'], default='None', help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig',		 action='store_true', default=False, help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi',  type=str, default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')

### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=20, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr',				 type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight',      type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')

#### from hipt
parser.add_argument('--model_type',     type=str, default='clam_sb', help='Type of model to use',
                    choices=['clam_sb', 'clam_mb', 'mil', 'dgcn', 'mi_fcn', 'dsmil', 'hipt_n', 'hipt_lgp', 'gene', 'mm_simple', 'mm_linear'])
parser.add_argument('--features',       type=str, default='vits_tcga_pancancer_dino', help='Which features to use', choices=['resnet50_trunc', 'vits_tcga_pancancer_dino'])
parser.add_argument('--path_input_dim', type=int, default=384, help='Size of patch embedding size (384 for DINO)')
parser.add_argument('--pretrain_4k',    type=str, default='vit4k_xs_dino', help='Whether to initialize the 4K Transformer in HIPT', choices=['None', 'vit4k_xs_dino'])
parser.add_argument('--pretrain_WSI',   type=str, default='None', help='Whether to initialize the WSI Transformer in HIPT', choices=['None', 'filler'])
parser.add_argument('--freeze_4k',      action='store_true', default=True, help='Whether to freeze the 4K Transformer in HIPT')
parser.add_argument('--freeze_WSI',     action='store_true', default=False, help='Whether to freeze the WSI Transformer in HIPT')

parser.add_argument('--gene_samples', type=int, default=1, help='Number of samples to take per patient')
parser.add_argument('--freeze_BERT',    action='store_true', default=False, help='Whether to freeze the BERT Mutation model')
parser.add_argument('--pretrain_BERT',  action='store_true', default=False, help='Whether to use pretrained BERT Mutation model')

parser.add_argument('--agg',     type=str, default='mean', help='Type of agg',
                    choices=['mean'])

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if 'hipt' in args.model_type:
    args.mode = 'pyramid'
    
if 'mm' in args.model_type:
    args.mode = 'multimodal'

### Creates Experiment Code from argparse + Folder Name to Save Results
args = get_custom_exp_code(args)
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
print("Experiment Name:", args.exp_code)

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
	import random
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
			'k_start': args.k_start,
			'k_end': args.k_end,
			'task': args.task,
			'max_epochs': args.max_epochs, 
			'results_dir': args.results_dir, 
			'lr': args.lr,
			'experiment': args.exp_code,
			'reg': args.reg,
			'label_frac': args.label_frac,
# 			'inst_loss': args.inst_loss,
			'bag_loss': args.bag_loss,
			'bag_weight': args.bag_weight,
			'seed': args.seed,
			'model_type': args.model_type,
			'model_size_wsi': args.model_size_wsi,
			'model_size_omic': args.model_size_omic,
			"use_drop_out": args.drop_out,
			'weighted_sample': args.weighted_sample,
			'gc': args.gc,
			'opt': args.opt}
print('\nLoad Dataset')
print(args.task)

study = "_".join(args.task.split('_')[:2])
if args.mode == 'pyramid' or args.mode == 'multimodal':
    study_dir = '{}/extracted_mag20x_patch4096_fp/{}_pt_patch_features_384'.format(study, args.features)
else:
    study_dir = '{}/extracted_mag20x_patch256_fp/{}_pt_patch_features'.format(study, args.features)
	
if args.task == 'tcga_brca_survival':
	args.n_classes = 4
	dataset = Generic_MIL_Survival_Dataset(csv_path='./dataset_csv/tcga_brca_subset.csv.zip',
										  data_dir=os.path.join(args.data_root_dir, study_dir),
										  mode = args.mode,
										   apply_sig = args.apply_sig,
										   shuffle = False, 
										   seed = args.seed, 
										   print_info = True,
										   patient_strat= False,
										   n_bins=4,
										   label_col = 'survival_months',
                                           gene_samples = args.gene_samples,
										   ignore=[])
	
# if 'survival' in args.task:
# 	args.n_classes = 4
# 	study = '_'.join(args.task.split('_')[:2])
# 	if study == 'tcga_kirc' or study == 'tcga_kirp':
# 		combined_study = 'tcga_kidney'
# 	elif study == 'tcga_luad' or study == 'tcga_lusc':
# 		combined_study = 'tcga_lung'
# 	else:
# 		combined_study = study
# 	study_dir = '%s_20x_features' % combined_study
# 	dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all_clean.csv.zip' % (args.dataset_path, combined_study),
# 										   mode = args.mode,
# 										   apply_sig = args.apply_sig,
# 										   data_dir= os.path.join(args.data_root_dir, study_dir),
# 										   shuffle = False, 
# 										   seed = args.seed, 
# 										   print_info = True,
# 										   patient_strat= False,
# 										   n_bins=4,
# 										   label_col = 'survival_months',
# 										   ignore=[])
else:
	raise NotImplementedError

# exit(0)

if isinstance(dataset, Generic_MIL_Survival_Dataset):
	args.task_type = 'survival'
else:
	raise NotImplementedError

### Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
	os.mkdir(args.results_dir)

### Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
	os.makedirs(args.results_dir)

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
	print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
	sys.exit()

### Sets the absolute path of split_dir
args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
print("split_dir", args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
	print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
	print("{}:  {}".format(key, val))        

if __name__ == "__main__":
	start = timer()
	results = main(args)
	end = timer()
	print("finished!")
	print("end script")
	print('Script Time: %f seconds' % (end - start))