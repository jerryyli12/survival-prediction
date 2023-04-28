from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
import random
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from utils.utils import generate_split, nth


class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv', mode = 'omic', apply_sig = False,
        shuffle = False, seed = 7, print_info = True, n_bins = 4, ignore=[],
        patient_strat=False, label_col = None, filter_dict = {}, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        slide_data = pd.read_csv(csv_path, low_memory=False)
        print(slide_data)
        #slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        import pdb
        #pdb.set_trace()

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})

        self.patient_dict = patient_dict
    
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes=len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:12]
        self.mode = mode
        self.cls_ids_prep()
                
        if print_info:
            self.summarize()

        ### Signatures
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./dataset_csv_sig/signatures.csv')
        else:
            self.signatures = None
            
        ### EDITS START HERE
        self.gene_samples = 1
        
        ### MUTATION STUFF
        self.min_mutation_nb = 20
        self.mutation_sampling_nb = 100
        self.vocab_size = 4
        self.mut_df = None
        
        if self.mode == 'mutation' or self.mode == 'multimodal':
            if os.path.isfile('mut_df.csv'):
                self.mut_df = pd.read_csv('mut_df.csv')
            else:
                mut_data_path = '../multimodal-histo-gene/pytorch-pretrained-BERT-master/data/Mutation/mut_data_gene2vec_brca.csv'
                mut_gene_list_path = '../multimodal-histo-gene/pytorch-pretrained-BERT-master/data/Mutation/mut_gene_list_gene2vec_brca.txt'
                mut_gene_list = np.loadtxt(mut_gene_list_path, dtype=str)

                if mut_data_path.endswith(".csv"):
                    df = pd.read_csv(mut_data_path, sep=",", index_col=0)
                elif mut_data_path.endswith(".pkl"):
                    df = pd.read_pickle(mut_data_path, compression='zip')
                    df = df.astype("category")
                else:
                    raise NotImplementedError

                if self.vocab_size == 4:
                    mutation_vocab = {
                        "CLS": 0,
                        "MASK": 3,
                        "Not_mutated": 1,
                        np.nan: np.nan,  # nan -> not measured => not sampled
                        "Missense_Mutation": 2,
                        "Silent": 2,
                        "3'UTR": 2,
                        "Nonsense_Mutation": 2,
                        "5'UTR": 2,
                        "Frame_Shift_Del": 2,
                        "Intron": 2,
                        "RNA": 2,
                        "Splice_Site": 2,
                        "In_Frame_Del": 2,
                        "Splice_Region": 2,
                        "Frame_Shift_Ins": 2,
                        "3'Flank": 2,
                        "5'Flank": 2,
                        "Translation_Start_Site": 2,
                        "Nonstop_Mutation": 2,
                        "In_Frame_Ins": 2,
                        "IGR": 2
                    }
                else:
                    mutation_vocab = {
                        "CLS": 0,
                        "MASK": 20,
                        "Not_mutated": 1,
                        np.nan: np.nan,  # nan -> not measured => not sampled
                        "Missense_Mutation": 2,
                        "Silent": 3,
                        "3'UTR": 4,
                        "Nonsense_Mutation": 5,
                        "5'UTR": 6,
                        "Frame_Shift_Del": 7,
                        "Intron": 8,
                        "RNA": 9,
                        "Splice_Site": 10,
                        "In_Frame_Del": 11,
                        "Splice_Region": 12,
                        "Frame_Shift_Ins": 13,
                        "3'Flank": 14,
                        "5'Flank": 15,
                        "Translation_Start_Site": 16,
                        "Nonstop_Mutation": 17,
                        "In_Frame_Ins": 18,
                        "IGR": 19
                    }

                gene_map = {gene: i+1 for i, gene in enumerate(mut_gene_list)}  # {'g1': 1, 'g2': 2, 'g3': 3, ...}
                # change patient name to patient id (from 0 to num_patient)
                if isinstance(df.index, pd.MultiIndex):
                    df.index = df.index.map(lambda x: '_'.join(map(str, x)))
                # patient_map = {patient_id: i for i, patient_id in enumerate(df.index.to_list())}
                df_t = df.T
                # map genes to gene code (coming from gene_map)
                df_t['gene_ids'] = df_t.index.map(gene_map)
                # get columns: gene_ids, patient_ids, mutation_ids
                self.mut_df = pd.melt(df_t, id_vars=['gene_ids'], var_name='patient_ids', value_name='mutation_ids')
                # replace patient name to patient indices
                self.mut_df['patient_ids'] = self.mut_df['patient_ids'].map(lambda x: x[:-3])
                # replace mutation name to mutation indices (coming from mutation_vocab)
                self.mut_df['mutation_ids'] = self.mut_df['mutation_ids'].map(mutation_vocab)
                self.mut_df.to_csv('mut_df.csv')
                
            print(self.mut_df)
            print(self.mut_df['mutation_ids'].value_counts())
        
        ### RNASEQ STUFF
        self.sampling_nb = 100
        self.num_quantiles = 5
        self.rna_df = None
        
        if self.mode == 'rnaseq' or self.mode == 'multimodal':
            if os.path.isfile('rna_df.csv'):
                self.rna_df = pd.read_csv('rna_df.csv')
            else:
                rna_data_path = '../multimodal-histo-gene/pytorch-pretrained-BERT-master/data/RNAseq/rna_data_gene2vec_brca.csv'
                rna_gene_list_path = '../multimodal-histo-gene/pytorch-pretrained-BERT-master/data/RNAseq/rna_gene_list_gene2vec_brca.txt'
                rna_gene_list = np.loadtxt(rna_gene_list_path, dtype=str)

                if rna_data_path.endswith(".csv"):
                    df = pd.read_csv(rna_data_path, sep=",", index_col=0)
                elif rna_data_path.endswith(".pkl"):
                    df = pd.read_pickle(rna_data_path, compression='zip')
                    df = df.astype("float32")
                else:
                    raise NotImplementedError

                bins_cut = [-float("inf"), -0.84, -0.25, 0.25, 0.84, float("inf")]
                labels_list = list(range(1, self.num_quantiles + 1))
                pd.options.mode.chained_assignment = None
                for i, gene in enumerate(df.columns):
                    print(i, gene)
                    df[gene] = pd.cut(df[gene], bins=bins_cut, labels=labels_list)

                gene_map = {gene: i+1 for i, gene in enumerate(rna_gene_list)}  # {'g1': 1, 'g2': 2, 'g3': 3, ...}
                # if multiple index -> join to 1
                if isinstance(df.index, pd.MultiIndex):
                    df.index = df.index.map(lambda x: '_'.join(map(str, x)))
                # change patient name to patient id (from 0 to num_patient)
                # patient_map = {patient_id: i for i, patient_id in enumerate(df.index.to_list())}
                df_t = df.T
                # map genes to gene code (coming from gene_map)
                df_t['gene_ids'] = df_t.index.map(gene_map)
                # get columns: gene_ids, patient_ids, rnaseq_bin
                self.rna_df = pd.melt(df_t, id_vars=['gene_ids'], var_name='patient_ids', value_name='rnaseq_bin')
                self.rna_df[['rnaseq_bin']] = self.rna_df[['rnaseq_bin']].astype(int)
                # replace patient name with patient indices
                self.rna_df['patient_ids'] = self.rna_df['patient_ids'].map(lambda x: x[:-3])
                self.rna_df.to_csv('rna_df.csv')
                
            print(self.rna_df)


    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]


    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}


    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))


    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode, signatures=self.signatures, data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes, mut_df=self.mut_df, rna_df = self.rna_df, min_mutation_nb=self.min_mutation_nb, mutation_sampling_nb=self.mutation_sampling_nb, sampling_nb=self.sampling_nb, gene_samples=self.gene_samples, split_key=split_key)
        else:
            split = None
        
        return split


    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            test_split = None #self.get_split_from_df(all_splits=all_splits, split_key='test')

            ### --> Normalizing Data
#             print("****** Normalizing Data ******")
#             scalers = train_split.get_scaler()
#             train_split.apply_scaler(scalers=scalers)
#             val_split.apply_scaler(scalers=scalers)
            #test_split.apply_scaler(scalers=scalers)
            ### <--
        return train_split, val_split#, test_split


    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def __getitem__(self, idx):
        return None


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, gene_samples=1, mode: str='omic', **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(mode=mode, **kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False
        self.gene_samples = gene_samples

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        
        c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        
        if not self.use_h5:
            if self.mode == 'mutation':
                df_item_og = self.mut_df[self.mut_df['patient_ids'] == case_id]
                all_samples = []
                for i in range(self.gene_samples):
                    df_item = df_item_og.copy()
                    if self.mutation_sampling_nb < df_item.shape[0]:
                        # check nb of mutation (i.e. not nan = 1) for this patient
                        n_mut = int(np.sum(df_item['mutation_ids'] > 1))
                        if n_mut < self.min_mutation_nb:
                            # select all mutations not nan
                            mut_rows = df_item[df_item['mutation_ids'] > 1]
                        else:
                            # random nb of mut btw min_mutation_nb and min(mutation_sampling_nb and n_mut)
                            n_mut = random.randint(self.min_mutation_nb, min(self.mutation_sampling_nb, n_mut))
                            mut_rows = df_item[df_item['mutation_ids'] > 1].sample(n=n_mut)
                        # rest of non nan mut
                        n_mut_nan = self.mutation_sampling_nb - n_mut
                        non_mut_rows = df_item[df_item['mutation_ids'] == 1].sample(n=n_mut_nan)
                        if n_mut > 0:
                            df_item = pd.concat([mut_rows, non_mut_rows])
                        else:
                            df_item = non_mut_rows
                    # return tuple of longtensors (gene_inds, mutation_inds) with CLS at the beginning
                    tokens = torch.LongTensor(df_item[['gene_ids', 'mutation_ids']].values)
                    all_samples.append(torch.cat((torch.LongTensor([0, 0]).unsqueeze(dim=0), tokens), dim=0))
                return torch.stack(all_samples), torch.zeros((1,1)), label, event_time, c
            elif self.mode == 'rnaseq':
                df_item_og = self.rna_df[self.rna_df['patient_ids'] == case_id]
                all_samples = []
                for i in range(self.gene_samples):
                    df_item = df_item_og.copy()
                    if self.sampling_nb < df_item.shape[0]:
                        df_item = df_item[df_item['rnaseq_bin'] > 0].sample(n=self.sampling_nb, )
                    # return tuple of longtensors (gene_inds, mutation_inds) with CLS at the beginning
                    tokens = torch.LongTensor(df_item[['gene_ids', 'rnaseq_bin']].values)
                    all_samples.append(torch.cat((torch.LongTensor([0, 0]).unsqueeze(dim=0), tokens), dim=0))
                return torch.stack(all_samples), torch.zeros((1,1)), label, event_time, c
            elif self.mode == 'pyramid':
                path_features = []
                for slide_id in slide_ids:
                    full_path = os.path.join(data_dir, '{}.pt'.format(slide_id.replace(".svs","")))
                    try:
                        path_features.append(torch.load(full_path))
                    except:
                        print("yikes")
                        path_features.append(torch.zeros((1, 192)))
                path_features = torch.cat(path_features, dim=0)
                return path_features.unsqueeze(dim=0), torch.zeros((1,1)), label, event_time, c
        else:
            return None


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, mode, mut_df, rna_df, min_mutation_nb, mutation_sampling_nb, sampling_nb, gene_samples, split_key, signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.mut_df = mut_df
        self.rna_df = rna_df
        self.min_mutation_nb = min_mutation_nb
        self.mutation_sampling_nb = mutation_sampling_nb
        self.sampling_nb = sampling_nb
        if split_key == 'train':
            self.gene_samples = 1
        else:
            self.gene_samples = gene_samples
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)
