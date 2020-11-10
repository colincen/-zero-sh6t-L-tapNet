#!/usr/bin/env python
import torch
from typing import Tuple, List
from models.modules.context_embedder_base import ContextEmbedderBase
from models.modules.emission_scorer_base import EmissionScorerBase
from models.modules.transition_scorer import TransitionScorerBase
from models.modules.seq_labeler import SequenceLabeler
from models.modules.conditional_random_field import ConditionalRandomField
from utils.iter_helper import pad_tensor    

class ZeroShotSeqLabeler(torch.nn.Module):
    def __init__(self,
                 opt,
                 context_embedder=None,
                 emission_scorer=None,
                 decoder=None,
                 config=None,
                 emb_log=None):
        super(ZeroShotSeqLabeler, self).__init__()
        self.opt =opt
        self.context_embedder = context_embedder
        self.emission_scorer = emission_scorer
        self.decoder = decoder
        self.config = config
        self.emb_log = emb_log
    
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
        
        

        emission = self.emission_scorer(token_reps, token_masks, pad_slot_names_reps, \
                                        pad_slot_names_mask,pad_slot_vals_reps, \
                                        pad_slot_vals_mask, label_ids)
        
        # batch_size x seq_len x emb_size  
        # batch_size x seq_len 
        # batch_size x label_size x emb_size
        # batch_size x label_size
        # batch_size x label_size x val_num x emb_size
        # batch_size x label_size x val_num

        logits = emission

        label_mask = (torch.zeros(label_ids.size(), device=label_ids.device).type_as(label_ids) == label_ids)
        label_mask = (label_mask == 0)
        label_mask = label_mask.byte()


        # print(label_ids)
        label_ids = torch.nn.functional.relu(label_ids - 1)
        loss, prediction = torch.FloatTensor([0]).to(label_ids.device), None



        # print(label_ids)
        # print(label_mask)

        if self.training:
            loss = self.decoder.forward(logits=logits,
                                        tags=label_ids,
                                        mask=label_mask)

        else:
            prediction = self.decoder.decode(logits=logits, masks=label_mask)
        

        if self.training:
            return loss
        else:
            return prediction
    
    