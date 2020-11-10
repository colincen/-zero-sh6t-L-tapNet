#!/usr/bin/env python
import torch
from typing import Tuple, List
from models.modules.context_embedder_base import ContextEmbedderBase
from models.modules.emission_scorer_base import EmissionScorerBase
from models.modules.transition_scorer import TransitionScorerBase
from models.modules.seq_labeler import SequenceLabeler
from models.modules.conditional_random_field import ConditionalRandomField

class ZeroShotSeqLabeler(torch.nn.Module):
    def __init__(self,
                 opt,
                 context_embedder=None,
                 emission_scorer=None):
        super(ZeroShotSeqLabeler, self).__init__()
        self.opt =opt
        self.context_embedder = context_embedder
        self.emission_scorer = emission_scorer
    
    def forward(
        self,
        token_ids,
        slot_names,
        slot_names_mask,
        slot_vals,
        slot_vals_mask,
        label_ids):

        token_reps, token_masks, pad_slot_names_reps, \
        pad_slot_names_mask,pad_slot_vals_reps, \ 
        pad_slot_vals_mask = self.context_embedder(token_ids, slot_names,\
                                    slot_names_mask, slot_vals, slot_vals_mask)
        

        # batch_size x seq_len x emb_size  
        # batch_size x seq_len 
        # batch_size x label_size x emb_size
        # batch_size x label_size x val_num x emb_size
        # batch_size x label_size x val_num



        return 0
    