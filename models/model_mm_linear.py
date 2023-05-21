import torch
import torch.nn as nn
from models.model_hierarchical_mil import HIPT_LGP_FC
from models.model_gene import GENE_FC
from models.model_utils import *

import os
import random

import sys
sys.path.append('../multimodal-histo-gene/pytorch-pretrained-BERT-master')
from pytorch_pretrained_bert import *
sys.path.append('../HIPT_4K/')
from vision_transformer4k import vit4k_xs

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
    
def load_ckpt(ckpt_path: str, model: BertModel):
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


class MM_LINEAR(nn.Module):
    def __init__(self, path_input_dim=384, size_arg = "small", dropout=0.25, n_classes=4,
     pretrain_4k='None', freeze_4k=False, pretrain_WSI='None', freeze_WSI=False, freeze_BERT=False, pretrain_BERT=False):
        super(MM_LINEAR, self).__init__()
        
        ### WSI
        
        self.size_dict_path = {"small": [384, 192, 192], "big": [1024, 512, 384]}
        #self.fusion = fusion
        size = self.size_dict_path[size_arg]

        ### Local Aggregation
        self.local_vit = vit4k_xs()
        if pretrain_4k != 'None':
            print("Loading Pretrained Local VIT model...",)
            state_dict = torch.load('../HIPT_4K/Checkpoints/%s.pth' % pretrain_4k, map_location='cpu')['teacher']
            state_dict = {k.replace('module.', ""): v for k, v in state_dict.items()}
            state_dict = {k.replace('backbone.', ""): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = self.local_vit.load_state_dict(state_dict, strict=False)
            print("Done!")
        if freeze_4k:
            print("Freezing Pretrained Local VIT model")
            for param in self.local_vit.parameters():
                param.requires_grad = False
            print("Done")

        ### Global Aggregation
        self.pretrain_WSI = pretrain_WSI
        if pretrain_WSI != 'None':
            pass
        else:
            self.global_phi = nn.Sequential(nn.Linear(192, 192), nn.ReLU(), nn.Dropout(0.25))
            self.global_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=192, nhead=3, dim_feedforward=192, dropout=0.25, activation='relu'
                ), 
                num_layers=2
            )
            self.global_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, n_classes=1)
            self.global_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25)])

        ### MUT
    
        config_dir = '../multimodal-histo-gene/pytorch-pretrained-BERT-master/configs'
        
        print('Using Mutation model')
        config_name = 'config_mutation_final'
        ckpt_path = '../multimodal-histo-gene/pytorch-pretrained-BERT-master/models/mutation/best_model/floral-sweep-1_fold_0.pt'
        gene_list_path = '../multimodal-histo-gene/pytorch-pretrained-BERT-master/data/Mutation/mut_gene_list_gene2vec_brca.txt'
            
        gene_list = np.loadtxt(gene_list_path, dtype=str)
        
        config = BertConfig.from_json_file(os.path.join(config_dir, config_name+".json"))
        config.rnaseq = False
        set_all_seed(seed=config.seed)
        model = BertModel(config, gene_list)

        if pretrain_BERT:
            print("Loading pretrained BERT model")
            model = load_ckpt(ckpt_path, model)
        if freeze_BERT:
            print("Freezing BERT")
            for param in model.parameters():
                param.requires_grad = False
            print("Done")
        
        self.mut_model = model
        
        ### RNA

        print('Using RNASeq model')
        config_name = 'config_mutation_final'  # this probably needs to be fixed
        ckpt_path = '../multimodal-histo-gene/pytorch-pretrained-BERT-master/models/rna_seq/best_model/logical-sweep-1_fold_0.pt'
        gene_list_path = '../multimodal-histo-gene/pytorch-pretrained-BERT-master/data/RNAseq/rna_gene_list_gene2vec_brca.txt'
            
        gene_list = np.loadtxt(gene_list_path, dtype=str)
        
        config = BertConfig.from_json_file(os.path.join(config_dir, config_name+".json"))
        config.rnaseq = True
        set_all_seed(seed=config.seed)
        model = BertModel(config, gene_list)
        
        if pretrain_BERT:
            print("Loading pretrained BERT model")
            model = load_ckpt(ckpt_path, model)
        if freeze_BERT:
            print("Freezing BERT")
            for param in model.parameters():
                param.requires_grad = False
            print("Done")
        
        self.rna_model = model
        
        self.classifier = nn.Linear(size[1] + 200 + 200, n_classes)
        

    def forward(self, wsi, mut, rna):
        h_4096 = wsi
        ### Global
        if self.pretrain_WSI != 'None':
            h_WSI = self.global_vit(h_4096.unsqueeze(dim=0))
        else:
            h_4096 = self.global_phi(h_4096)
            h_4096 = self.global_transformer(h_4096.unsqueeze(1)).squeeze(1)
            A_4096, h_4096 = self.global_attn_pool(h_4096)  
            A_4096 = torch.transpose(A_4096, 1, 0)
            A_4096 = F.softmax(A_4096, dim=1) 
            h_path = torch.mm(A_4096, h_4096)
            h_WSI = self.global_rho(h_path)
        
        if mut.dim() == 2:
            mut = mut.unsqueeze(0)
        _, mut_pooled = self.mut_model(mut)
        
        if rna.dim() == 2:
            rna = rna.unsqueeze(0)
        _, rna_pooled = self.rna_model(rna)
        
        x = torch.cat([h_WSI, mut_pooled, rna_pooled], dim=1)
        logits = self.classifier(x)
        
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat, None, None
    
    def relocate(self):
        print('Relocating')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.local_vit = nn.DataParallel(self.local_vit, device_ids=device_ids).to('cuda:0')
            if self.pretrain_WSI != 'None':
                self.global_vit = nn.DataParallel(self.global_vit, device_ids=device_ids).to('cuda:0')

        if self.pretrain_WSI == 'None':
            self.global_phi = self.global_phi.to(device)
            self.global_transformer = self.global_transformer.to(device)
            self.global_attn_pool = self.global_attn_pool.to(device)
            self.global_rho = self.global_rho.to(device)

        self.classifier = self.classifier.to(device)
        self.mut_model = self.mut_model.to(device)
        self.rna_model = self.rna_model.to(device)
        