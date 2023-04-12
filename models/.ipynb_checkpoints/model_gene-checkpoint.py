import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import random
import numpy as np
import os
from os.path import join
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *
import sys
sys.path.append('../multimodal-histo-gene/pytorch-pretrained-BERT-master')
from pytorch_pretrained_bert import *
# from tests.train_bert_mutation import set_all_seed

def set_all_seed(seed: int = 42) -> None:
    """
    For reproducibility
    :param seed: seed number
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load_ckpt(ckpt_path: str, model: BertModel, optimizer: BertAdam):
    """
    load a model weights
    :param ckpt_path:
    :param model:
    :param optimizer:
    :return: model, optimizer and epoch
    """
    ckpt = torch.load(ckpt_path)
    model.load_state_dict({k.replace('bert.', ''): v for k,v in ckpt['state_dict'].items()}, strict=False)
    return model
    
class GENE_FC(nn.Module):
    def __init__(self, dropout=0.25, n_classes=4):
        super(GENE_FC, self).__init__()
        config_dir = '../multimodal-histo-gene/pytorch-pretrained-BERT-master/configs'
        config_name = 'config_mutation_final'
        ckpt_path = '../multimodal-histo-gene/pytorch-pretrained-BERT-master/models/best_model/mutation_best_model_10k.pt'
        
        gene_list_path = '../multimodal-histo-gene/Mutation/data/gene_list_brca_final.txt'
        gene_list = np.loadtxt(gene_list_path, dtype=str)
        
        config = BertConfig.from_json_file(os.path.join(config_dir, config_name+".json"))
        set_all_seed(seed=config.seed)
        model = BertModel(config, gene_list)

        # Initialise Optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=config.warmup_proportion,)
        
        model = load_ckpt(ckpt_path, model, optimizer)
        
        print(model)
        self.model = model
        self.cls = nn.Linear(200, n_classes)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, pooled = self.model(x)
        x = self.cls(pooled)
#         if x.dim() == 3:
#             x = x.squeeze(0)
        logits = x
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat, None, None
    
    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.cls = self.cls.to(device)
        