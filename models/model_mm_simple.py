import torch
import torch.nn as nn
from models.model_hierarchical_mil import HIPT_LGP_FC
from models.model_gene import GENE_FC

class MM_SIMPLE(nn.Module):
    def __init__(self, agg='mean', split=0):
        super(MM_SIMPLE, self).__init__()
        self.agg = agg
        
        self.wsi_model = HIPT_LGP_FC()
        self.wsi_model.relocate()
        self.wsi_model.load_state_dict(torch.load('results_OLD/5foldcv/_nll_surv_a0.0_5foldcv_gc32/tcga_brca__nll_surv_a0.0_5foldcv_gc32_s1/s_{}_checkpoint.pt'.format(split)))
        # self.wsi_model.load_state_dict(torch.load('results/5foldcv/_nll_surv_a0.0_5foldcv_gc32/tcga_brca__nll_surv_a0.0_5foldcv_gc32_s1/s_{}_checkpoint.pt'.format(split)))
        
        self.mut_model = GENE_FC(mode='mutation')
        self.mut_model.relocate()
        self.mut_model.load_state_dict(torch.load('results/5foldcv/GENE20_mutation_pretrain_nll_surv_a0.0_lr1e-04_5foldcv_gc32/tcga_brca_GENE20_mutation_pretrain_nll_surv_a0.0_lr1e-04_5foldcv_gc32_s1/s_{}_checkpoint.pt'.format(split)))
        
        self.rna_model = GENE_FC(mode='rnaseq')
        self.rna_model.relocate()
        self.rna_model.load_state_dict(torch.load('results/5foldcv/GENE20_rnaseq_pretrain_nll_surv_a0.0_lr1e-04_5foldcv_gc32/tcga_brca_GENE20_rnaseq_pretrain_nll_surv_a0.0_lr1e-04_5foldcv_gc32_s1/s_{}_checkpoint.pt'.format(split)))
        
        print('Fully loaded on split', split)
        

    def forward(self, wsi, mut, rna):
        w_h, w_s, w_y, _, _ = self.wsi_model(wsi)
        m_h, m_s, m_y, _, _ = self.mut_model(mut)
        r_h, r_s, r_y, _, _ = self.rna_model(rna)
        
        # MAX, MEDIAN, MEAN
        if self.agg == 'mean':
            # S = (w_s + m_s + r_s) / 3
            S = (w_s + m_s) / 2

        # logits = self.classifier(h_WSI)
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]
        # hazards = torch.sigmoid(logits)
        # S = torch.cumprod(1 - hazards, dim=1)
        return None, S, None, None, None

    
    # def relocate(self):
    #     print('Relocating')
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.wsi_model = self.wsi_model.relocate()
    #     self.mut_model = self.mut_model.relocate()
    #     self.rna_model = self.rna_model.relocate()