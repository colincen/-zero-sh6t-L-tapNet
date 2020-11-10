#!/usr/bin/env python
import torch
import logging
import sys
from transformers import BertModel, ElectraModel
from torchnlp.word_to_vector import GloVe
import torch.nn.utils.rnn as rnn_utils


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


class ContextEmbedderBase(torch.nn.Module):
    def __init__(self):
        super(ContextEmbedderBase, self).__init__()

    def forward(self, *args, **kwargs) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        :param args:
        :param kwargs:
        :return: test_token_reps, support_token_reps, test_sent_reps, support_sent_reps
        """
        raise NotImplementedError()


class BertContextEmbedder(ContextEmbedderBase):
    def __init__(self, opt):
        super(BertContextEmbedder, self).__init__()
        self.opt = opt
        self.embedder = self.build_embedder()

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            support_token_ids: torch.Tensor = None,
            support_segment_ids: torch.Tensor = None,
            support_nwp_index: torch.Tensor = None,
            support_input_mask: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor):
        """
        get context representation
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len, 1)
        :param test_input_mask: (batch_size, test_len)

        ======= Support features ======
        We allow to only embed query to enable single sentence embedding, but such feature is NOT used now.
        (Separate embedding is achieved through special sub classes)

        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len, 1)
        :param support_input_mask: (batch_size, support_size, support_len)
        :return:
            if do concatenating representation:
                return (test_reps, support_reps, test_sent_reps, support_sent_reps):
                    test_reps, support_reps:  token reps (batch_size, support_size, nwp_sent_len, emb_len)
                    test_sent_reps, support_sent_reps: sent reps (batch_size, support_size, 1, emb_len)
            else do representation for a single sent (No support staff):
                return test_reps, shape is (batch_size, nwp_sent_len, emb_len)
        """
        if support_token_ids is not None:
            return self.concatenate_reps(
                test_token_ids, test_segment_ids, test_nwp_index, test_input_mask,
                support_token_ids, support_segment_ids, support_nwp_index, support_input_mask,
            )
        else:
            return self.single_reps(test_token_ids, test_segment_ids, test_nwp_index, test_input_mask,)

    def build_embedder(self):
        """ load bert here """
        return BertModel.from_pretrained(self.opt.bert_path)

    def concatenate_reps(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        """ get token reps of a sent pair. """
        support_size = support_token_ids.shape[1]
        test_len = test_token_ids.shape[-1] - 2  # max len, exclude [CLS] and [SEP] token
        support_len = support_token_ids.shape[-1] - 1  # max len, exclude [SEP] token
        batch_size = support_token_ids.shape[0]
        ''' expand test input to shape: (batch_size, support_size, test_len)'''
        test_token_ids, test_segment_ids, test_input_mask, test_nwp_index = self.expand_test_item(
            test_token_ids, test_segment_ids, test_input_mask, test_nwp_index, support_size)
        ''' concat test and support '''
        input_ids = self.cat_test_and_support(test_token_ids, support_token_ids)
        segment_ids = self.cat_test_and_support(test_segment_ids, support_segment_ids)
        input_mask = self.cat_test_and_support(test_input_mask, support_input_mask)
        ''' flatten input '''
        input_ids, segment_ids, input_mask = self.flatten_input(input_ids, segment_ids, input_mask)
        test_nwp_index, support_nwp_index = self.flatten_index(test_nwp_index), self.flatten_index(support_nwp_index)
        ''' get concat reps '''
        sequence_output = self.embedder(input_ids, input_mask, segment_ids)[0]
        ''' extract reps '''
        # select pure sent part, remove [SEP] and [CLS], notice: seq_len1 == seq_len2 == max_len.
        test_reps = sequence_output.narrow(-2, 1, test_len)  # shape:(batch, test_len, rep_size)
        support_reps = sequence_output.narrow(-2, 2 + test_len, support_len)  # shape:(batch, support_len, rep_size)
        # select non-word-piece tokens' representation
        nwp_test_reps = self.extract_non_word_piece_reps(test_reps, test_nwp_index)
        nwp_support_reps = self.extract_non_word_piece_reps(support_reps, support_nwp_index)
        # resize to shape (batch_size, support_size, sent_len, emb_len)
        reps_size = nwp_test_reps.shape[-1]
        nwp_test_reps = nwp_test_reps.view(batch_size, support_size, -1, reps_size)
        nwp_support_reps = nwp_support_reps.view(batch_size, support_size, -1, reps_size)
        test_reps = test_reps.view(batch_size, support_size, -1, reps_size)
        support_reps = support_reps.view(batch_size, support_size, -1, reps_size)
        # get whole sent reps
        test_sent_reps = self.get_sent_reps(test_reps, test_input_mask)
        support_sent_reps = self.get_sent_reps(support_reps, support_input_mask)
        return nwp_test_reps, nwp_support_reps, test_sent_reps, support_sent_reps

    def single_reps(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        """ get token reps of a single sent. """
        test_len = test_token_ids.shape[-1] - 2  # max len, exclude [CLS] and [SEP] token
        batch_size = test_token_ids.shape[0]
        ''' get bert reps '''
        test_sequence_output = self.embedder(test_token_ids, test_input_mask, test_segment_ids)[0]
        ''' extract reps '''
        # select pure sent part, remove [SEP] and [CLS], notice: seq_len1 == seq_len2 == max_len.
        test_reps = test_sequence_output.narrow(-2, 1, test_len)  # shape:(batch, test_len, rep_size)
        # select non-word-piece tokens' representation
        nwp_test_reps = self.extract_non_word_piece_reps(test_reps, test_nwp_index)
        # get whole word reps, unsuqeeze to fit interface
        test_sent_reps = self.get_sent_reps(test_reps.unsqueeze(1), test_input_mask.unsqueeze(1)).squeeze(1)
        return nwp_test_reps, test_sent_reps

    def get_sent_reps(self, reps, input_mask):
        """
         Average token reps to get a whole sent reps
        :param reps:   (batch_size, support_size, sent_len, emb_len)
        :param input_mask:  (batch_size, support_size, sent_len)
        :return:  averaged reps (batch_size, support_size, sent_len, emb_len)
        """
        batch_size, support_size, sent_len, reps_size = reps.shape
        mask_len = input_mask.shape[-1]
        # count each sent's tokens, to avoid over div with pad,  shape: (batch_size * support_size, 1)
        token_counts = torch.sum(input_mask.contiguous().view(-1, mask_len), dim=1).unsqueeze(-1)
        sp_token_num = input_mask.shape[-1] - reps.shape[-2]  # num of [CLS], [SEP] tokens
        token_counts = token_counts - sp_token_num + 0.00001  # calculate pure token num and remove zero
        # mask pad-token's reps to 0 vectors [Notice that by default pad token's reps are not 0-vector]
        if sp_token_num == 2:
            trimed_mask = input_mask.narrow(-1, 1, reps.shape[-2]).float()  # remove mask of [CLS], [SEP]
        elif sp_token_num == 1:
            trimed_mask = input_mask.narrow(-1, 0, reps.shape[-2]).float()  # remove mask of [SEP]
        else:
            raise RuntimeError("Unexpected sp_token_num.")
        reps = reps * trimed_mask.unsqueeze(-1)
        # sum reps, shape (batch_size * support_size, emb_len)
        sum_reps = torch.sum(reps.contiguous().view(-1, sent_len, reps_size), dim=1)
        # averaged reps (batch_size, support_size, emb_len)
        ave_reps = torch.div(sum_reps, token_counts.float()).contiguous().view(batch_size, support_size, reps_size)
        return ave_reps.unsqueeze(-2)

    def expand_test_item(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_input_mask: torch.Tensor,
            test_nwp_index: torch.Tensor,
            support_size: int,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        return self.expand_it(test_token_ids, support_size), self.expand_it(test_segment_ids, support_size), \
               self.expand_it(test_input_mask, support_size), self.expand_it(test_nwp_index, support_size)

    def expand_it(self, item: torch.Tensor, support_size):
        expand_shape = list(item.unsqueeze_(1).shape)
        expand_shape[1] = support_size
        return item.expand(expand_shape)

    def cat_test_and_support(self, test_item, support_item):
        return torch.cat([test_item, support_item], dim=-1)

    def flatten_input(self, input_ids, segment_ids, input_mask):
        """ resize shape (batch_size, support_size, cat_len) to shape (batch_size * support_size, sent_len) """
        sent_len = input_ids.shape[-1]
        input_ids = input_ids.view(-1, sent_len)
        segment_ids = segment_ids.view(-1, sent_len)
        input_mask = input_mask.view(-1, sent_len)
        return input_ids, segment_ids, input_mask

    def flatten_index(self, nwp_index):
        """ resize shape (batch_size, support_size, index_len, 1) to shape (batch_size * support_size, index_len, 1) """
        nwp_sent_len = nwp_index.shape[-2]
        return nwp_index.contiguous().view(-1, nwp_sent_len, 1)

    def extract_non_word_piece_reps(self, reps, index):
        """
        Use the first word piece as entire word representation
        As we have only one index for each token, we need to expand to the size of reps dim.
        """
        expand_shape = list(index.shape)
        expand_shape[-1] = reps.shape[-1]  # expend index over embedding dim
        index = index.expand(expand_shape)
        nwp_reps = torch.gather(input=reps, index=index, dim=-2)  # extract over token level
        return nwp_reps


class BertSchemaContextEmbedder(BertContextEmbedder):
    def __init__(self, opt):
        super(BertSchemaContextEmbedder, self).__init__(opt)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            support_token_ids: torch.Tensor = None,
            support_segment_ids: torch.Tensor = None,
            support_nwp_index: torch.Tensor = None,
            support_input_mask: torch.Tensor = None,
            reps_type: str = 'test_support',
    ) -> (torch.Tensor, torch.Tensor):
        """
        get context representation
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len, 1)
        :param test_input_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len, 1)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param reps_type: select the reps type, default: reps for test and support tokens. Special choice is for label
        :return:
            if do concatenating representation:
                return (test_reps, support_reps, test_sent_reps, support_sent_reps):
                    test_reps, support_reps:  token reps (batch_size, support_size, nwp_sent_len, emb_len)
                    test_sent_reps, support_sent_reps: sent reps (batch_size, support_size, 1, emb_len)
            else do representation for a single sent (No support staff):
                return test_reps, shape is (batch_size, nwp_sent_len, emb_len)
        """
        if reps_type == 'test_support':
            if support_token_ids is not None:
                return self.concatenate_reps(
                    test_token_ids, test_segment_ids, test_nwp_index, test_input_mask,
                    support_token_ids, support_segment_ids, support_nwp_index, support_input_mask,
                )
            else:
                return self.single_reps(test_token_ids, test_segment_ids, test_nwp_index, test_input_mask,)
        elif reps_type == 'label':
            return self.get_label_reps(test_token_ids, test_segment_ids, test_nwp_index, test_input_mask)

    def get_label_reps(self, test_token_ids, test_segment_ids, test_nwp_index, test_input_mask):
        batch_size = test_token_ids.shape[0]
        if self.opt.label_reps == 'cat':
            # todo: use label mask to represent a label with only in domain info
            reps = self.single_reps(test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, )
        elif self.opt.label_reps in ['sep', 'sep_sum']:
            input_ids, segment_ids, input_mask = self.flatten_input(test_token_ids, test_segment_ids,
                                                                    test_input_mask)
            # nwp_index = self.flatten_index(test_nwp_index)
            # get flatten reps: shape (batch_size * label_num, label_des_len)
            sequence_output = self.embedder(input_ids, input_mask, segment_ids)[0]
            reps_size = sequence_output.shape[-1]
            if self.opt.label_reps == 'sep':  # use cls as each label's reps
                # re-shape to  (batch_size, label_num, label_des_len)
                reps = sequence_output.narrow(-2, 0, 1)  # fetch all [CLS] shape:(batch, 1, rep_size)
                reps = reps.contiguous().view(batch_size, -1, reps_size)
            elif self.opt.label_reps == 'sep_sum':  # average all label reps as reps
                reps = sequence_output
                emb_mask = self.expand_mask(test_input_mask, 2, reps_size)
                # todo: use mask to get sum of single embedding
                raise NotImplementedError
            else:
                raise ValueError("Wrong label_reps choice ")
        else:
            raise ValueError("Wrong reps_type choice")
        return reps

    def expand_mask(self, item: torch.Tensor, expand_size, dim):
        new_item = item.unsqueeze(dim)
        expand_shape = list(new_item.shape)
        expand_shape[dim] = expand_size
        return new_item.expand(expand_shape)


class NormalContextEmbedder(ContextEmbedderBase):
    def __init__(self, opt, num_token):
        super(NormalContextEmbedder, self).__init__()
        self.opt = opt
        ''' load bert '''
        self.embedding_layer = torch.nn.Embedding(num_token, opt.emb_dim, padding_idx=0)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        """
        get context representation
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :return: (test_reps, support_reps, None, None):
            all their shape are (batch_size, support_size, nwp_sent_len, emb_len)
        """
        support_size = support_token_ids.shape[1]
        test_len = test_token_ids.shape[-1] - 2  # max len, exclude [CLS] and [SEP] token
        support_len = support_token_ids.shape[-1] - 1  # max len, exclude [SEP] token
        batch_size = support_token_ids.shape[0]
        ''' expand test input to shape: (batch_size, support_size, test_len)'''
        test_token_ids, test_segment_ids, test_input_mask, test_nwp_index = self.expand_test_item(
            test_token_ids, test_segment_ids, test_input_mask, test_nwp_index, support_size)

        ''' get reps '''
        test_reps = self.embedding_layer(test_token_ids)
        support_reps = self.embedding_layer(support_token_ids)

        return test_reps, support_reps, None, None

    def load_embedding(self):
        word2id = self.opt.word2id
        logging.info('Load embedding from pytorch-nlp.')
        if self.opt.embedding_cache:
            embedding_dict = GloVe(name = '6B', cache=self.opt.embedding_cache)  # load embedding cache from a specific place
        else:
            embedding_dict = GloVe()  # load embedding cache from local dir or download now
        logging.info('Load embedding finished.')
        self.embedding_layer.weight.data.uniform_(-0.25, 0.25)
        word_in_dict = 0
        # print(word2id)
        for word, idx in word2id.items():
            # if word in embedding_dict.stoi:
            if word in embedding_dict:
                word_in_dict += 1
            self.embedding_layer.weight.data[idx] = embedding_dict[word]

        logging.info('Word embedding size: {0}'.format(self.embedding_layer.weight.data.size()))
        logging.info('unk word {}'.format(word_in_dict / len(word2id)))

    def expand_test_item(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_input_mask: torch.Tensor,
            test_nwp_index: torch.Tensor,
            support_size: int,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        return self.expand_it(test_token_ids, support_size), self.expand_it(test_segment_ids, support_size), \
               self.expand_it(test_input_mask, support_size), self.expand_it(test_nwp_index, support_size)

    def expand_it(self, item: torch.Tensor, support_size):
        expand_shape = list(item.unsqueeze_(1).shape)
        expand_shape[1] = support_size
        return item.expand(expand_shape)

    def flatten_input(self, input_ids):
        """ resize shape (batch_size, support_size, sent_len) to shape (batch_size * support_size, sent_len) """
        sent_len = input_ids.shape[-1]
        input_ids = input_ids.view(-1, sent_len)
        return input_ids


class BilstmContextEmbedder(NormalContextEmbedder):
    def __init__(self, opt, num_token):
        super(BilstmContextEmbedder, self).__init__(opt, num_token)
        self.BilstmEncoder = torch.nn.LSTM(input_size=opt.emb_dim,
                                            hidden_size= int (opt.hidden_size / 2),
                                            bias=True,
                                            batch_first=True,
                                            bidirectional=True)
        self.Dropout = torch.nn.Dropout(0.5)
    
    def forward(self,
                token_ids,
                slot_names,
                slot_names_mask,
                slot_vals,
                slot_vals_mask,
                ):
        """
        token_ids: (batch_size x seq_len)
        slot_names: (batch_size x label_size x max_name_len)
        slot_names_mask: (batch_size x label_size x max_name_len)        
        slot_vals: (batch_size x label_size x val_num x max_val_len)
        slot_vals_mask: (batch_size x label_size x val_num x max_val_len)
        """
        # print('ok')

        batch_size = token_ids.size(0)
        label_size = slot_names_mask.size(1)
        val_num = slot_vals.size(2)

        token_masks = (torch.zeros(token_ids.size()).type_as(token_ids) == token_ids)
        token_masks = (token_masks == 0)
        token_masks = token_masks.byte()

        token_length = torch.sum(token_masks, 1)


        token_reps = self.embedding_layer(token_ids)


        token_packed = rnn_utils.pack_padded_sequence(token_reps, token_length, batch_first=True, enforce_sorted=False)
        token_reps, _ = self.BilstmEncoder(token_packed)
        token_reps, _ = rnn_utils.pad_packed_sequence(token_reps, batch_first=True)
        token_reps = self.Dropout(token_reps)
        # print(token_reps.size())


        slot_names_merge = slot_names.view(batch_size * label_size, -1)
        slot_names_mask_merge = slot_names_mask.view(batch_size * label_size, -1)
        slot_names_length = torch.sum(slot_names_mask_merge, 1)

        slot_names_index = torch.arange(0, slot_names_length.size(0), device=token_ids.device)

        slot_names_index = slot_names_index[slot_names_length > 0]
        domain_slot_names_index = slot_names_index.long()

        slot_names_merge = slot_names_merge[domain_slot_names_index, :]
        
       



        pad_slot_names_mask = (slot_names_length > 0).long()




        slot_names_length = slot_names_length[domain_slot_names_index]
        

        slot_names_reps = self.embedding_layer(slot_names_merge)

        slot_names_packed = rnn_utils.pack_padded_sequence(slot_names_reps, slot_names_length, batch_first=True, enforce_sorted=False)
        slot_names_reps, _ = self.BilstmEncoder(slot_names_packed)
        slot_names_reps, _ = rnn_utils.pad_packed_sequence(slot_names_reps, batch_first=True)
        slot_names_reps = self.Dropout(slot_names_reps)
        
        #######################
        slot_names_reps = torch.sum(slot_names_reps, -2)

        '''
        several ways
        * sum the embedding
        * mean the sum of embedding
        * l2 norm the mean
        '''
        # print(slot_names_length)
        # print(slot_names_reps.size())   

        # print(slot_names_reps)
        slot_names_reps = torch.div(slot_names_reps, slot_names_length.unsqueeze(-1))
        # print(slot_names_reps)
        ################################

        
        output_dim = slot_names_reps.size(1)
        pad_slot_names_reps = torch.zeros(size=(batch_size * label_size, output_dim), device=token_ids.device)
        pad_slot_names_reps[domain_slot_names_index, :] = slot_names_reps
        
        pad_slot_names_reps = pad_slot_names_reps.view(batch_size, label_size, -1)
        pad_slot_names_mask = pad_slot_names_mask.view(batch_size, label_size)
        


        slot_vals_merge = slot_vals.view(batch_size * label_size * val_num, -1)
        slot_vals_mask_merge = slot_vals_mask.view(batch_size * label_size * val_num, -1)
        slot_vals_length = torch.sum(slot_vals_mask_merge, -1)
        slot_vals_index = torch.arange(0, slot_vals_length.size(0), device=token_ids.device)
        slot_vals_index = slot_vals_index[slot_vals_length > 0]
        domain_slot_vals_index = slot_vals_index.long()

        slot_vals_merge = slot_vals_merge[domain_slot_vals_index, :]


        pad_slot_vals_mask = (slot_vals_length > 0).long()


        slot_vals_length = slot_vals_length[domain_slot_vals_index]
        slot_vals_reps = self.embedding_layer(slot_vals_merge)

        slot_vals_packed = rnn_utils.pack_padded_sequence(slot_vals_reps, slot_vals_length, batch_first=True, enforce_sorted=False)
        slot_vals_reps, _ = self.BilstmEncoder(slot_vals_packed)
        slot_vals_reps, _ = rnn_utils.pad_packed_sequence(slot_vals_reps, batch_first=True)
        slot_vals_reps = self.Dropout(slot_vals_reps)

        ####################################

        slot_vals_reps = torch.sum(slot_vals_reps, -2)
        
        '''
        several ways
        * sum the embedding
        * mean the sum of embedding
        * l2 norm the mean
        '''
        # print(slot_vals_reps)
        slot_vals_reps = torch.div(slot_vals_reps, slot_vals_length.unsqueeze(-1))
        # print(slot_vals_reps)
        # print(slot_vals_reps.size())

        #####################################

        output_dim = slot_vals_reps.size(1)
        pad_slot_vals_reps = torch.zeros(size=(batch_size * label_size * val_num, output_dim), device=token_ids.device)
        pad_slot_vals_reps[domain_slot_vals_index, :] = slot_vals_reps

        pad_slot_vals_reps = pad_slot_vals_reps.view(batch_size, label_size, val_num, -1)
        pad_slot_vals_mask = pad_slot_vals_mask.view(batch_size, label_size, val_num)

        return token_reps, token_masks, pad_slot_names_reps, \
                pad_slot_names_mask, pad_slot_vals_reps, pad_slot_vals_mask
        

class BertSeparateContextEmbedder(BertContextEmbedder):
    def __init__(self, opt):
        super(BertSeparateContextEmbedder, self).__init__(opt)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            support_token_ids: torch.Tensor = None,
            support_segment_ids: torch.Tensor = None,
            support_nwp_index: torch.Tensor = None,
            support_input_mask: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor):
        """
        get context representation
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len, 1)
        :param test_input_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len, 1)
        :param support_input_mask: (batch_size, support_size, support_len)
        :return: if do concatenating representation:
                return (test_reps, support_reps, test_sent_reps, support_sent_reps):
                    test_reps, support_reps:  token reps (batch_size, support_size, nwp_sent_len, emb_len)
                    test_sent_reps, support_sent_reps: sent reps (batch_size, support_size, 1, emb_len)
            else do representation for a single sent (No support staff):
                return test_reps, shape is (batch_size, nwp_sent_len, emb_len)
        """
        if support_token_ids is not None:
            return self.separate_reps(
                test_token_ids, test_segment_ids, test_nwp_index, test_input_mask,
                support_token_ids, support_segment_ids, support_nwp_index, support_input_mask,
            )
        else:
            return self.single_reps(test_token_ids, test_segment_ids, test_nwp_index, test_input_mask,)

    def separate_reps(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            support_token_ids: torch.Tensor = None,
            support_segment_ids: torch.Tensor = None,
            support_nwp_index: torch.Tensor = None,
            support_input_mask: torch.Tensor = None,
            reps_type: str = None,
    ) -> (torch.Tensor, torch.Tensor):
        """ Separately get two sent reps. """
        support_size = support_token_ids.shape[1]
        test_len = test_token_ids.shape[-1] - 2  # max len, exclude [CLS] and [SEP] token
        support_len = support_token_ids.shape[-1] - 1  # max len, exclude [SEP] token
        batch_size = support_token_ids.shape[0]
        ''' flatten input '''
        support_token_ids, support_segment_ids, support_input_mask = self.flatten_input(
            support_token_ids, support_segment_ids, support_input_mask)
        support_nwp_index = self.flatten_index(support_nwp_index)
        ''' get bert reps '''
        test_sequence_output = self.embedder(test_token_ids, test_input_mask, test_segment_ids)[0]
        support_sequence_output = self.embedder(support_token_ids, support_input_mask, support_segment_ids)[0]
        ''' extract reps '''
        # select pure sent part, remove [SEP] and [CLS], notice: seq_len1 == seq_len2 == max_len.
        test_reps = test_sequence_output.narrow(-2, 1, test_len)  # shape:(batch, test_len, rep_size)
        support_reps = support_sequence_output.narrow(-2, 1, support_len)  # shape:(batch * support_size, support_len, rep_size)
        # select non-word-piece tokens' representation
        nwp_test_reps = self.extract_non_word_piece_reps(test_reps, test_nwp_index)
        nwp_support_reps = self.extract_non_word_piece_reps(support_reps, support_nwp_index)
        # resize to shape (batch_size, support_size, sent_len, emb_len)
        reps_size = test_reps.shape[-1]
        nwp_test_reps = self.expand_it(nwp_test_reps, support_size).contiguous()
        nwp_support_reps = nwp_support_reps.view(batch_size, support_size, -1, reps_size).contiguous()
        # get whole sent reps
        test_sent_reps = self.get_sent_reps(test_reps, test_input_mask)
        support_sent_reps = self.get_sent_reps(support_reps, support_input_mask)
        return nwp_test_reps, nwp_support_reps, test_sent_reps, support_sent_reps


class BertSchemaSeparateContextEmbedder(BertSeparateContextEmbedder, BertSchemaContextEmbedder):
    def __init__(self, opt):
        super(BertSchemaSeparateContextEmbedder, self).__init__(opt)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            support_token_ids: torch.Tensor = None,
            support_segment_ids: torch.Tensor = None,
            support_nwp_index: torch.Tensor = None,
            support_input_mask: torch.Tensor = None,
            reps_type: str = 'test_support',
    ) -> (torch.Tensor, torch.Tensor):
        """
        get context representation
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len, 1)
        :param test_input_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len, 1)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param reps_type: fit schema method
        :return: if do concatenating representation:
                return (test_reps, support_reps, test_sent_reps, support_sent_reps):
                    test_reps, support_reps:  token reps (batch_size, support_size, nwp_sent_len, emb_len)
                    test_sent_reps, support_sent_reps: sent reps (batch_size, support_size, 1, emb_len)
            else do representation for a single sent (No support staff):
                return test_reps, shape is (batch_size, nwp_sent_len, emb_len)
        """
        batch_size = test_token_ids.shape[0]
        if reps_type == 'test_support':
            if support_token_ids is not None:
                return self.separate_reps(
                    test_token_ids, test_segment_ids, test_nwp_index, test_input_mask,
                    support_token_ids, support_segment_ids, support_nwp_index, support_input_mask,
                )
            else:
                return self.single_reps(test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, )
        elif reps_type == 'label':
            return self.get_label_reps(test_token_ids, test_segment_ids, test_nwp_index, test_input_mask)


class ElectraContextEmbedder(BertContextEmbedder):
    """ Electra based context embedder """
    def __init__(self, opt):
        super(ElectraContextEmbedder, self).__init__(opt)

    def build_embedder(self):
        """ Load pretrained params """
        return ElectraModel.from_pretrained(self.opt.bert_path)


class ElectraSchemaContextEmbedder(BertSchemaContextEmbedder):
    """ Electra based Context Embedder with schema info """
    def __init__(self, opt):
        super(ElectraSchemaContextEmbedder, self).__init__(opt)

    def build_embedder(self):
        """ Load pretrained params """
        return ElectraModel.from_pretrained(self.opt.bert_path)

