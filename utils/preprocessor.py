# coding:utf-8
import collections
from typing import List, Tuple, Dict
import torch
import pickle
import json
import os
from transformers import BertTokenizer
from utils.data_loader import FewShotExample, DataItem
from utils.iter_helper import pad_tensor
from utils.config import SP_TOKEN_NO_MATCH, SP_LABEL_O, SP_TOKEN_O


FeatureItem = collections.namedtuple(   # text or raw features
    "FeatureItem",
    [
        "tokens",  # tokens corresponding to input token ids, eg: word_piece tokens with [CLS], [SEP]
        "labels",  # labels for all input position, eg; label for word_piece tokens
        "data_item",
        "token_ids",
        "segment_ids",
        "nwp_index",
        "input_mask",
        "output_mask"
    ]
)

ModelInput = collections.namedtuple(  # digit features for computation
    "ModelInput",   # all element shape: test: (1, test_len) support: (support_size, support_len)
    [
        "token_ids",  # token index list
        "segment_ids",  # bert [SEP] ids
        "nwp_index",  # non-word-piece word index to extract non-word-piece tokens' reps (only useful for bert).
        "input_mask",  # [1] * len(sent), 1 for valid (tokens, cls, sep, word piece), 0 is padding in batch construction
        "output_mask",  # [1] * len(sent), 1 for valid output, 0 for padding, eg: 1 for original tokens in sl task
    ]
)

ZeroShotFeatureItem = collections.namedtuple(
    "ZeroShotFeatureItem",
    [
        "tokens",
        "labels",
        "slot_names",
        "slot_vals",
    ]
)
ZeroShotModelInput = collections.namedtuple(
    "ZeroShotModelInput",
    [
        "token_ids", 
        "slot_names", 
        "slot_names_mask",  
        "slot_vals", 
        "slot_vals_mask",
    ]
)

class FewShotFeature(object):
    """ pre-processed data for prediction """

    def __init__(
            self,
            gid: int,  # global id
            test_gid: int,
            batch_gid: int,
            test_input: ModelInput,
            test_feature_item: FeatureItem,
            support_input: ModelInput,
            support_feature_items: List[FeatureItem],
            test_target: torch.Tensor,  # 1) CRF, shape: (1, test_len)  2)SMS, shape: (support_size, t_len, s_len)
            support_target: torch.Tensor,  # 1) shape: (support_len, label_size)
            label_input=None,
            label_items=None,
    ):
        self.gid = gid
        self.test_gid = test_gid
        self.batch_gid = batch_gid
        ''' padded tensor for model '''
        self.test_input = test_input  # shape: (1, test_len)
        self.support_input = support_input  # shape: (support_size, support_len)
        # output:
        # 1)CRF, shape: (1, test_len)
        # 2)SMS, shape: (support_size, test_len, support_len)
        self.test_target = test_target
        self.support_target = support_target
        ''' raw feature '''
        self.test_feature_item = test_feature_item
        self.support_feature_items = support_feature_items
        self.label_input = label_input
        self.label_items = label_items

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.__dict__)


class InputBuilderBase:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, example, max_support_size, label2id
    ) -> (FeatureItem, ModelInput, List[FeatureItem], ModelInput):
        raise NotImplementedError

    def data_item2feature_item(self, data_item: DataItem, seg_id: int) -> FeatureItem:
        raise NotImplementedError

    def get_test_model_input(self, feature_item: FeatureItem) -> ModelInput:
        ret = ModelInput(
            token_ids=torch.LongTensor(feature_item.token_ids),
            segment_ids=torch.LongTensor(feature_item.segment_ids),
            nwp_index=torch.LongTensor(feature_item.nwp_index),
            input_mask=torch.LongTensor(feature_item.input_mask),
            output_mask=torch.LongTensor(feature_item.output_mask)
        )
        return ret

    def get_support_model_input(self, feature_items: List[FeatureItem], max_support_size: int) -> ModelInput:
        pad_id = self.tokenizer.vocab['[PAD]']
        token_ids = self.pad_support_set([f.token_ids for f in feature_items], pad_id, max_support_size)
        segment_ids = self.pad_support_set([f.segment_ids for f in feature_items], 0, max_support_size)
        nwp_index = self.pad_support_set([f.nwp_index for f in feature_items], [0], max_support_size)
        input_mask = self.pad_support_set([f.input_mask for f in feature_items], 0, max_support_size)
        output_mask = self.pad_support_set([f.output_mask for f in feature_items], 0, max_support_size)
        ret = ModelInput(
            token_ids=torch.LongTensor(token_ids),
            segment_ids=torch.LongTensor(segment_ids),
            nwp_index=torch.LongTensor(nwp_index),
            input_mask=torch.LongTensor(input_mask),
            output_mask=torch.LongTensor(output_mask)
        )
        return ret

    def pad_support_set(self, item_lst: List[List[int]], pad_value: int, max_support_size: int) -> List[List[int]]:
        """
        pre-pad support set to insure: 1. each spt set has same sent num 2. each sent has same length
        (do padding here because: 1. all support sent are considered as one tensor input  2. support set size is small)
        :param item_lst:
        :param pad_value:
        :param max_support_size:
        :return:
        """
        ''' pad sentences '''
        max_sent_len = max([len(x) for x in item_lst])  # max length among one
        ret = []
        for sent in item_lst:
            temp = sent[:]
            while len(temp) < max_sent_len:
                temp.append(pad_value)
            ret.append(temp)
        ''' pad support set size '''
        pad_item = [pad_value for _ in range(max_sent_len)]
        while len(ret) < max_support_size:
            ret.append(pad_item)
        return ret

    def digitizing_input(self, tokens: List[str], seg_id: int) -> (List[int], List[int]):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [seg_id for _ in range(len(tokens))]
        return token_ids, segment_ids

    def tokenizing(self, item: DataItem):
        """ Possible tokenizing for item """
        pass


class BertInputBuilder(InputBuilderBase):
    def __init__(self, tokenizer, opt):
        super(BertInputBuilder, self).__init__(tokenizer)
        self.opt = opt
        self.test_seg_id = 0
        self.support_seg_id = 0 if opt.context_emb == 'sep_bert' else 1  # 1 to cat support and query to get reps
        self.seq_ins = {}

    def __call__(self, example, max_support_size, label2id) -> (FeatureItem, ModelInput, List[FeatureItem], ModelInput):
        test_feature_item, test_input = self.prepare_test(example)
        support_feature_items, support_input = self.prepare_support(example, max_support_size)
        return test_feature_item, test_input, support_feature_items, support_input

    def prepare_test(self, example):
        test_feature_item = self.data_item2feature_item(data_item=example.test_data_item, seg_id=0)
        test_input = self.get_test_model_input(test_feature_item)
        return test_feature_item, test_input

    def prepare_support(self, example, max_support_size):
        support_feature_items = [self.data_item2feature_item(data_item=s_item, seg_id=self.support_seg_id) for s_item in
                                 example.support_data_items]
        support_input = self.get_support_model_input(support_feature_items, max_support_size)
        return support_feature_items, support_input

    def data_item2feature_item(self, data_item: DataItem, seg_id: int) -> FeatureItem:
        """ get feature_item for bert, steps: 1. do digitalizing 2. make mask """
        wp_mark, wp_text = self.tokenizing(data_item)
        if self.opt.task == 'sl':  # use word-level labels  [opt.label_wp is supported by model now.]
            labels = self.get_wp_label(data_item.seq_out, wp_text, wp_mark) if self.opt.label_wp else data_item.seq_out
        else:  # use sentence level labels
            labels = data_item.label
            if 'None' not in labels:
                # transfer label to index type such as `label_1`
                if self.opt.index_label:
                    labels = [self.opt.label2index_type[label] for label in labels]
                if self.opt.unused_label:
                    labels = [self.opt.label2unused_type[label] for label in labels]
        tokens = ['[CLS]'] + wp_text + ['[SEP]'] if seg_id == 0 else wp_text + ['[SEP]']
        token_ids, segment_ids = self.digitizing_input(tokens=tokens, seg_id=seg_id)
        nwp_index = self.get_nwp_index(wp_mark)
        input_mask = [1] * len(token_ids)
        output_mask = [1] * len(labels)   # For sl: it is original tokens; For sc: it is labels
        ret = FeatureItem(
            tokens=tokens,
            labels=labels,
            data_item=data_item,
            token_ids=token_ids,
            segment_ids=segment_ids,
            nwp_index=nwp_index,
            input_mask=input_mask,
            output_mask=output_mask,
        )
        return ret

    def get_nwp_index(self, word_piece_mark: list) -> torch.Tensor:
        """ get index of non-word-piece tokens, which is used to extract non-wp bert embedding in batch manner """
        return torch.nonzero(torch.LongTensor(word_piece_mark) - 1).tolist()  # wp mark word-piece with 1, so - 1

    def tokenizing(self, item: DataItem):
        """ Do tokenizing and get word piece data and get label on pieced words. """
        wp_text = self.tokenizer.wordpiece_tokenizer.tokenize(' '.join(item.seq_in))
        wp_mark = [int((len(w) > 2) and w[0] == '#' and w[1] == '#') for w in wp_text]  # mark wp as 1
        return wp_mark, wp_text

    def get_wp_label(self, label_lst, wp_text, wp_mark, label_pieced_words=False):
        """ get label on pieced words. """
        wp_label, label_idx = [], 0
        for ind, mark in enumerate(wp_mark):
            if mark == 0:  # label non-pieced token with original label
                wp_label.append(label_lst[label_idx])
                label_idx += 1  # pointer on non-wp labels
            elif mark == 1:  # label word-piece with whole word's label or with  [PAD] label
                pieced_label = wp_label[-1].replace('B-', 'I-') if label_pieced_words else '[PAD]'
                wp_label.append(pieced_label)
            if not wp_label[-1]:
                raise RuntimeError('Empty label')
        if not (len(wp_label) == len(wp_text) == len(wp_mark)):
            raise RuntimeError('ERROR: Failed to generate wp labels:{}{}{}{}{}{}{}{}{}{}{}'.format(
                len(wp_label), len(wp_text), len(wp_mark),
                '\nwp_lb', wp_label, '\nwp_text', wp_text, '\nwp_mk', wp_mark, '\nlabel', label_lst))


class SchemaInputBuilder(BertInputBuilder):
    def __init__(self, tokenizer, opt):
        super(SchemaInputBuilder, self).__init__(tokenizer, opt)

    def __call__(self, example, max_support_size, label2id) -> (FeatureItem, ModelInput, List[FeatureItem], ModelInput):
        test_feature_item, test_input = self.prepare_test(example)
        support_feature_items, support_input = self.prepare_support(example, max_support_size)
        if self.opt.label_reps in ['cat']:  # represent labels by concat all all labels
            label_input, label_items = self.prepare_label_feature(label2id)
            
        elif self.opt.label_reps in ['sep', 'sep_sum']:  # represent each label independently
            label_input, label_items = self.prepare_sep_label_feature(label2id)
        return test_feature_item, test_input, support_feature_items, support_input, label_items, label_input,

    def prepare_label_feature(self, label2id: dict):
        """ prepare digital input for label feature in concatenate style """
        text, wp_text, label, wp_label, wp_mark = [], [], [], [], []
        sorted_labels = sorted(label2id.items(), key=lambda x: x[1])
        for label_name, label_id in sorted_labels:
            if label_name == '[PAD]':
                continue
            tmp_text = self.convert_label_name(label_name)
            tmp_wp_text = self.tokenizer.tokenize(' '.join(tmp_text))
            text.extend(tmp_text)
            wp_text.extend(tmp_wp_text)
            label.extend(['O'] * len(tmp_text))
            wp_label.extend(['O'] * len(tmp_wp_text))
            wp_mark.extend([0] + [1] * (len(tmp_wp_text) - 1))
        label_item = self.data_item2feature_item(DataItem(text, label, wp_text, wp_label, wp_mark), 0)
       
        label_input = self.get_test_model_input(label_item)
       
        return label_input, label_item

    def prepare_sep_label_feature(self, label2id):
        """ prepare digital input for label feature separately """
        label_items = []
        for label_name in label2id:
            if label_name == '[PAD]':
                continue
            seq_in = self.convert_label_name(label_name)
            seq_out = ['None'] * len(seq_in)
            label = ['None']
            label_items.append(self.data_item2feature_item(DataItem(seq_in, seq_out, label), 0))
        label_input = self.get_support_model_input(label_items, len(label2id) - 1)  # no pad, so - 1
        return label_input, label_items

    def convert_label_name(self, name):
        text = []
        tmp_name = name
        if 'B-' in name:
            text.append('begin')
            tmp_name = name.replace('B-', '')
        elif 'I-' in name:
            text.append('inner')
            tmp_name = name.replace('I-', '')
        elif 'O' == name:
            text.append('ordinary')
            tmp_name = ''

        # special processing to label name
        name_translations = [('PER', 'person'), ('ORG', 'organization'), ('LOC', 'location'),
                             ('MISC', 'miscellaneous'), ('GPE', 'geographical political'),
                             ('NORP', 'nationalities or religious or political groups'),
                             # toursg data
                             ("ACK", "acknowledgment, as well as common expressions used for grounding"),
                             # ("CANCEL", "cancelation"),
                             # ("CLOSING", "closing remarks"),
                             # ("COMMIT", "commitment"),
                             # ("CONFIRM", "confirmation"),
                             # ("ENOUGH", "no more information is needed"),
                             # ("EXPLAIN", "an explanation/justification of a previous stated idea"),
                             # ("HOW_MUCH", "money or time amounts"),
                             # ("HOW_TO", "used to request/give specific instructions"),
                             ("INFO", "information request"),
                             # ("NEGATIVE", "negative responses"),
                             # ("OPENING", "opening remarks"),
                             # ("POSITIVE", "positive responses"),
                             # ("PREFERENCE", "preferences"),
                             # ("RECOMMEND", "recommendations"),
                             # ("THANK", "thank you remarks"),
                             # ("WHAT", "concept related utterances"),
                             # ("WHEN", "time related utterances"),
                             # ("WHERE", "location related utterances"),
                             # ("WHICH", "entity related utterances"),
                             # ("WHO", "person related utterances and questions"),
                             ]
        if tmp_name:
            for shot, long in name_translations:
                if tmp_name == shot:
                    text.append(long)
                    tmp_name = ''
                    break
        if tmp_name:
            text.extend(tmp_name.lower().split('_'))
        return text


class NormalInputBuilder(InputBuilderBase):
    def __init__(self, tokenizer):
        super(NormalInputBuilder, self).__init__(tokenizer)

    def __call__(self, example, max_support_size, label2id) -> (FeatureItem, ModelInput, List[FeatureItem], ModelInput):
        test_feature_item = self.data_item2feature_item(data_item=example.test_data_item, seg_id=0)
        test_input = self.get_test_model_input(test_feature_item)
        support_feature_items = [self.data_item2feature_item(data_item=s_item, seg_id=1) for s_item in
                                 example.support_data_items]
        support_input = self.get_support_model_input(support_feature_items, max_support_size)
        return test_feature_item, test_input, support_feature_items, support_input

    def data_item2feature_item(self, data_item: DataItem, seg_id: int) -> FeatureItem:
        """ get feature_item for bert, steps: 1. do padding 2. do digitalizing 3. make mask """
        tokens, labels = data_item.seq_in, data_item.seq_out
        token_ids, segment_ids = self.digitizing_input(tokens=tokens, seg_id=seg_id)
        nwp_index = [[i] for i in range(len(token_ids))]
        input_mask = [1] * len(token_ids)
        output_mask = [1] * len(data_item.seq_in)
        ret = FeatureItem(
            tokens=tokens,
            labels=labels,
            data_item=data_item,
            token_ids=token_ids,
            segment_ids=segment_ids,
            nwp_index=nwp_index,
            input_mask=input_mask,
            output_mask=output_mask,
        )
        return ret


class OutputBuilderBase:
    """  Digitalizing the output targets"""
    def __init__(self):
        pass

    def __call__(self, test_feature_item: FeatureItem, support_feature_items: FeatureItem,
                 label2id: dict, max_support_size: int):
        raise NotImplementedError

    def pad_support_set(self, item_lst: List[List[int]], pad_value: int, max_support_size: int) -> List[List[int]]:
        """
        pre-pad support set to insure: 1. each set has same sent num 2. each sent has same length
        (do padding here because: 1. all support sent are considered as one tensor input  2. support set size is small)
        :param item_lst:
        :param pad_value:
        :param max_support_size:
        :return:
        """
        ''' pad sentences '''
        max_sent_len = max([len(x) for x in item_lst])
        ret = []
        for sent in item_lst:
            temp = sent[:]
            while len(temp) < max_sent_len:
                temp.append(pad_value)
            ret.append(temp)
        ''' pad support set size '''
        pad_item = [pad_value for _ in range(max_sent_len)]
        while len(ret) < max_support_size:
            ret.append(pad_item)
        return ret


class FewShotOutputBuilder(OutputBuilderBase):
    """  Digitalizing the output targets as label id for non word piece tokens  """
    def __init__(self):
        super(FewShotOutputBuilder, self).__init__()

    def __call__(self, test_feature_item: FeatureItem, support_feature_items: FeatureItem,
                 label2id: dict, max_support_size: int):
        test_target = self.item2label_ids(test_feature_item, label2id)
        # to estimate emission, the support target is one-hot here
        support_target = [self.item2label_onehot(f_item, label2id) for f_item in support_feature_items]
        support_target = self.pad_support_set(support_target, self.label2onehot('[PAD]', label2id), max_support_size)
        return torch.LongTensor(test_target), torch.LongTensor(support_target)

    def item2label_ids(self, f_item: FeatureItem, label2id: dict):
        return [label2id[lb] for lb in f_item.labels]

    def item2label_onehot(self, f_item: FeatureItem, label2id: dict):
        return [self.label2onehot(lb, label2id) for lb in f_item.labels]

    def label2onehot(self, label: str, label2id: dict):
        onehot = [0 for _ in range(len(label2id))]
        onehot[label2id[label]] = 1
        return onehot


class SpecialOutputBuilder(OutputBuilderBase):
    def __init__(self, label2id, id2label):
        super(SpecialOutputBuilder, self).__init__(label2id, id2label)
        raise NotImplementedError


class FeatureConstructor:
    """
    Class for build feature and label2id dict
    Main function:
        construct_feature
        make_dict
    """
    def __init__(
            self,
            input_builder: InputBuilderBase,
            output_builder: OutputBuilderBase,
    ):
        self.input_builder = input_builder
        self.output_builder = output_builder

    def construct_feature(
            self,
            examples: List[FewShotExample],
            max_support_size: int,
            label2id: dict,
            id2label: dict,
    ) -> List[FewShotFeature]:
        all_features = []
        for example in examples:
            feature = self.example2feature(example, max_support_size, label2id, id2label)
            all_features.append(feature)
        return all_features

    def example2feature(
            self,
            example: FewShotExample,
            max_support_size: int,
            label2id: dict,
            id2label: dict
    ) -> FewShotFeature:
        test_feature_item, test_input, support_feature_items, support_input = self.input_builder(
            example, max_support_size, label2id)
        test_target, support_target = self.output_builder(
            test_feature_item, support_feature_items, label2id, max_support_size)
        ret = FewShotFeature(
            gid=example.gid,
            test_gid=example.test_id,
            batch_gid=example.batch_id,
            test_input=test_input,
            test_feature_item=test_feature_item,
            support_input=support_input,
            support_feature_items=support_feature_items,
            test_target=test_target,
            support_target=support_target,
        )
        return ret


class SchemaFeatureConstructor(FeatureConstructor):
    def __init__(
            self,
            input_builder: InputBuilderBase,
            output_builder: OutputBuilderBase,
    ):
        super(SchemaFeatureConstructor, self).__init__(input_builder, output_builder)

    def example2feature(
            self,
            example: FewShotExample,
            max_support_size: int,
            label2id: dict,
            id2label: dict
    ) -> FewShotFeature:
        test_feature_item, test_input, support_feature_items, support_input, label_items, label_input = \
            self.input_builder(example, max_support_size, label2id)
        
        


        test_target, support_target = self.output_builder(
            test_feature_item, support_feature_items, label2id, max_support_size)
        ret = FewShotFeature(
            gid=example.gid,
            test_gid=example.test_id,
            batch_gid=example.batch_id,
            test_input=test_input,
            test_feature_item=test_feature_item,
            support_input=support_input,
            support_feature_items=support_feature_items,
            test_target=test_target,
            support_target=support_target,
            label_input=label_input,
            label_items=label_items,
        )
        return ret




class NormalInputBuilderForZeroShot(object):
    def __init__(self, tokenizer, opt):
        self.tokenizer = tokenizer
        self.opt = opt
            
    def __call__(self, example, label2id, max_val_len=10) -> (ZeroShotFeatureItem, ZeroShotModelInput):
        # print(example.input_item.seq_in)
        utterance = self.prepare_utterance(example.input_item.seq_in)
        # print(utterance)
        slot_names, slot_names_mask = self.prepare_slot_name(example.input_item.slot_names, label2id)
        # print(slot_names)

        # print(slot_names_mask)
        slot_vals, slot_vals_mask = self.prepare_slot_val(example.input_item.slot_names, example.input_item.slot_vals, label2id, max_val_len)
        ret =ZeroShotModelInput(
            token_ids=torch.LongTensor(utterance),
            slot_names=torch.LongTensor(slot_names),
            slot_names_mask=torch.LongTensor(slot_names_mask),
            slot_vals=torch.LongTensor(slot_vals),
            slot_vals_mask=torch.LongTensor(slot_vals_mask)
        )
        return ret

    def prepare_utterance(self, seq_in):
        utterance = self.tokenizer.convert_tokens_to_ids(seq_in)
        return utterance

    def prepare_slot_name(self, slot_names, label2id):
        slot_names_B = ['B-' + i for i in slot_names]
        slot_names_I = ['I-' + i for i in slot_names]
        slot_names = slot_names_B + slot_names_I
        # mask = [0] * len(label2id)
        res_slot_names = []
        # print(slot_names)
        id2label = {v : k for k, v in label2id.items()}
        for idx in range(len(id2label)):
            name = id2label[idx]
            if name in slot_names or name == 'O' or name == '[PAD]':
                # mask[idx] = 1
                res_slot_names.append(self.convert_label_name(name))
            else:
                res_slot_names.append(['[PAD]'])
        res_slot_names, masks =  self.pad(res_slot_names, max_len=6, Type='one_layer')



        res_slot_names_ids = []
        for i in res_slot_names:
            temp = self.tokenizer.convert_tokens_to_ids(i)
            res_slot_names_ids.append(temp)

        # print(res_slot_names)
        # print(res_slot_names_ids)
        # print('-'*30)

        # print(len(res_slot_names_ids))
        # print(len(masks))
        return res_slot_names_ids, masks
       
    def prepare_slot_val(self, slot_names, slot_vals, label2id, max_val_len):
        assert len(slot_names) == len(slot_vals)
        id2label = {v:k for k,v in label2id.items()}
        slot2val = {}
        for i, name in enumerate(slot_names):
            slot2val[name] = slot_vals[i]
        slot_names_B = ['B-' + i for i in slot_names]
        slot_names_I = ['I-' + i for i in slot_names]
        slot_names = slot_names_B + slot_names_I

        
        res_slot_vals = [None] * len(label2id)
        res_val_masks = [None] * len(label2id)
        for id1 in range(len(id2label)):
            
            for id2 in range(len(slot_names)):
                temp_val = []
                if id2label[id1] == slot_names[id2]:
                    # print(slot2val[id2label[id1][2:]])

                    if id2label[id1][0] == 'B':
                        temp_val = self.add_prefix_to_val(prefix='begin', val_list= slot2val[id2label[id1][2:]])
                    elif id2label[id1][0] == 'I':
                        temp_val = self.add_prefix_to_val(prefix='inner', val_list= slot2val[id2label[id1][2:]])
                    temp_val, temp_mask = self.pad(temp_val, max_len=max_val_len, Type='one_layer')
                    res_slot_vals[id1] = temp_val
                    res_val_masks[id1] = temp_mask

        for i, line in enumerate(res_slot_vals):
            if line == None:
                pad_list = ['[PAD]'] * max_val_len
                pad_list = [pad_list] * len(slot_vals[0])
                pad_mask = [0] * max_val_len
                pad_mask = [pad_mask] * len(slot_vals[0])
                res_slot_vals[i] = pad_list
                res_val_masks[i] = pad_mask


        
        res_slot_val_ids = []
        for i in res_slot_vals:
            temp1 = []
            for j in i:

                temp1.append(self.tokenizer.convert_tokens_to_ids(j))
            res_slot_val_ids.append(temp1)

        
        return res_slot_val_ids, res_val_masks


            # if name in slot_names or name == 'O' or name == '[PAD]':
            #     # mask[idx] = 1
            #     res_slot_names.append(self.convert_label_name(name))
            # else:
            #     res_slot_names.append(['[PAD]'])
        # for i in slot_vals:
        #     temp = self.pad(i, 10, 'one_layer')
        #     res_slot_vals.append(temp)


            



         


        

    def add_prefix_to_val(self, prefix, val_list):

        res= []
        for i in val_list:
            temp= []
            temp.append(prefix)
            temp.extend(i)
            res.append(temp)

        return res

    def convert_label_name(self, name):
        text = []
        tmp_name = name
        if 'B-' in name:
            text.append('begin')
            tmp_name = name.replace('B-', '')
        elif 'I-' in name:
            text.append('inner')
            tmp_name = name.replace('I-', '')
        elif 'O' == name:
            text.append('ordinary')
            tmp_name = ''

        # special processing to label name
        name_translations = [('PER', 'person'), ('ORG', 'organization'), ('LOC', 'location'),
                             ('MISC', 'miscellaneous'), ('GPE', 'geographical political'),
                             ('NORP', 'nationalities or religious or political groups'),
                             # toursg data
                             ("ACK", "acknowledgment, as well as common expressions used for grounding"),
                             ("poi", "point of information"),
                             # ("CANCEL", "cancelation"),
                             # ("CLOSING", "closing remarks"),
                             # ("COMMIT", "commitment"),
                             # ("CONFIRM", "confirmation"),
                             # ("ENOUGH", "no more information is needed"),
                             # ("EXPLAIN", "an explanation/justification of a previous stated idea"),
                             # ("HOW_MUCH", "money or time amounts"),
                             # ("HOW_TO", "used to request/give specific instructions"),
                             ("INFO", "information request"),
                             # ("NEGATIVE", "negative responses"),
                             # ("OPENING", "opening remarks"),
                             # ("POSITIVE", "positive responses"),
                             # ("PREFERENCE", "preferences"),
                             # ("RECOMMEND", "recommendations"),
                             # ("THANK", "thank you remarks"),
                             # ("WHAT", "concept related utterances"),
                             # ("WHEN", "time related utterances"),
                             # ("WHERE", "location related utterances"),
                             # ("WHICH", "entity related utterances"),
                             # ("WHO", "person related utterances and questions"),
                             ]
        if tmp_name:
            for shot, long in name_translations:
                if tmp_name == shot:
                    text.append(long)
                    tmp_name = ''
                    break
        if tmp_name:
            if tmp_name == '[PAD]':
              text.extend([tmp_name])
            else:    
                text.extend(tmp_name.lower().split('_'))
        return text

    def pad(self, l, max_len = -1, Type = 'one_layer'):
        masks = []
        if Type == 'one_layer':
            for i in range(len(l)):
                while(len(l[i]) < max_len):
                    l[i].append('[PAD]')
                l[i] = l[i][:max_len]
            for i in l:
                temp = []
                for j in i:
                    if j == '[PAD]':
                        temp.append(0)
                    else:
                        temp.append(1)
                masks.append(temp)
        elif Type == 'two_layers':
            pass

        return l, masks


class NormalOutputBuilderForZeroShot(object):
    






def flatten(l):
    """ convert list of list to list"""
    return [item for sublist in l for item in sublist]


def make_dict(opt, examples) -> (Dict[str, int], Dict[int, str]):
    """
    make label2id dict
    label2id must follow rules:
    For sequence labeling:
        1. id(PAD)=0 id(O)=1  2. id(B-X)=i  id(I-X)=i+1
    For (multi-label) text classification:
        1. id(PAD)=0
    """
    def purify(l):
        """ remove B- and I- """
        return set([item.replace('B-', '').replace('I-', '') for item in l])

    ''' collect all label from: all test set & all support set '''
    all_labels = []
    label2id = {}
    for example in examples:
        if opt.task == 'sl':
            all_labels.append(example.test_data_item.seq_out)
            all_labels.extend([data_item.seq_out for data_item in example.support_data_items])
        elif opt.task == 'zero-shot':
            all_labels.append(example.input_item.seq_out)
        else:
            all_labels.append(example.test_data_item.label)
            all_labels.extend([data_item.label for data_item in example.support_data_items])
    ''' collect label word set '''
    label_set = sorted(list(purify(set(flatten(all_labels)))))  # sort to make embedding id fixed

    # transfer label to index type such as `label_1`
    if opt.index_label:
        if 'label2index_type' not in opt:
            opt.label2index_type = {}
            for idx, label in enumerate(label_set):
                opt.label2index_type[label] = 'label_' + str(idx)
        else:
            max_label_idx = max([int(value.replace('label_', '')) for value in opt.label2index_type.values()])
            for label in label_set:
                if label not in opt.label2index_type:
                    max_label_idx += 1
                    opt.label2index_type[label] = 'label_' + str(max_label_idx)
        label_set = [opt.label2index_type[label] for label in label_set]
    elif opt.unused_label:
        if 'label2unused_type' not in opt:
            opt.label2unused_type = {}
            for idx, label in enumerate(label_set):
                opt.label2unused_type[label] = '[unused' + str(idx) + ']'
        else:
            max_label_idx = max([int(value.replace('[unused', '').replace(']', '')) for value in opt.label2unused_type.values()])
            for label in label_set:
                if label not in opt.label2unused_type:
                    max_label_idx += 1
                    opt.label2unused_type[label] = '[unused' + str(max_label_idx) + ']'
        label_set = [opt.label2unused_type[label] for label in label_set]
    else:
        pass
    ''' build dict '''
    label2id['[PAD]'] = len(label2id)  # '[PAD]' in first position and id is 0
    if opt.task == 'sl' or opt.task == 'zero-shot':
        label2id['O'] = len(label2id)
        for label in label_set:
            if label == 'O':
                continue
            label2id['B-' + label] = len(label2id)
            label2id['I-' + label] = len(label2id)
    else:  # sc
        for label in label_set:
            label2id[label] = len(label2id)
    ''' reverse the label2id '''
    id2label = dict([(idx, label) for label, idx in label2id.items()])
    return label2id, id2label


def make_word_dict(all_files: List[str]) -> (Dict[str, int], Dict[int, str]):
    all_words = []
    word2id = {}
    for file in all_files:
        with open(file, 'r') as reader:
            raw_data = json.load(reader)
        if 'slots' not in raw_data.keys() or 'slot_vals' not in raw_data.keys():
            for domain_n, domain in raw_data.items():
            # Notice: the batch here means few shot batch, not training batch
                for batch_id, batch in enumerate(domain):
                    all_words.extend(batch['support']['seq_ins'])
                    all_words.extend(batch['query']['seq_ins'])
            word_set = sorted(list(set(flatten(all_words))))  # sort to make embedding id fixed

        else:
            for line in raw_data['seq_in']:
                # print(line)
                all_words.extend(line)
            for line in raw_data['slots']:
                all_words.extend(line)
            all_words.append('begin')
            all_words.append('inner')
            all_words.append('ordinary')
            word_set = sorted(list(set(all_words)))  # sort to make embedding id fixed
            
    
    # word_set = sorted(list(set(flatten(all_words))))  # sort to make embedding id fixed
    for word in ['[PAD]', '[OOV]'] + word_set:
        word2id[word] = len(word2id)
    id2word = dict([(idx, word) for word, idx in word2id.items()])
    return word2id, id2word


def make_mask(token_ids: torch.Tensor, label_ids: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    input_mask = (token_ids != 0).long()
    output_mask = (label_ids != 0).long()  # mask
    return input_mask, output_mask


def save_feature(path, features, label2id, id2label):
    with open(path, 'wb') as writer:
        saved_features = {
            'features': features,
            'label2id': label2id,
            'id2label': id2label,
        }
        pickle.dump(saved_features, writer)


def load_feature(path):
    with open(path, 'rb') as reader:
        saved_feature = pickle.load(reader)
        return saved_feature['features'], saved_feature['label2id'], saved_feature['id2label']


def make_preprocessor(opt):
    """ make preprocessor """
    transformer_style_embs = ['bert', 'sep_bert', 'electra']

    ''' select input_builder '''
    if opt.context_emb not in transformer_style_embs:
        word2id, id2word = make_word_dict([opt.train_path, opt.dev_path, opt.test_path])
        opt.word2id = word2id


    if opt.context_emb in transformer_style_embs:
        tokenizer = BertTokenizer.from_pretrained(opt.bert_vocab)
        if opt.use_schema:
            
            input_builder = SchemaInputBuilder(tokenizer=tokenizer, opt=opt)
        else:
            input_builder = BertInputBuilder(tokenizer=tokenizer, opt=opt)

    elif opt.context_emb == 'elmo':
        raise NotImplementedError
    elif opt.context_emb in ['glove', 'raw']:
        tokenizer = MyTokenizer(word2id=word2id, id2word=id2word)
        input_builder = NormalInputBuilder(tokenizer=tokenizer)
    elif opt.context_emb == 'bilstm':
        tokenizer = MyTokenizer(word2id=word2id, id2word=id2word)
        input_builder = NormalInputBuilderForZeroShot(opt=opt, tokenizer=tokenizer)

    else:
        raise TypeError('wrong word representation type')

    return input_builder

    # ''' select output builder '''
    # output_builder = FewShotOutputBuilder()

    # ''' build preprocessor '''
    # if opt.use_schema:
    #     preprocessor = SchemaFeatureConstructor(input_builder=input_builder, output_builder=output_builder)
    # else:
    #     preprocessor = FeatureConstructor(input_builder=input_builder, output_builder=output_builder)
    # return preprocessor


def make_label_mask(opt, path, label2id):
    """ disable cross domain transition """
    label_mask = [[0] * len(label2id) for _ in range(len(label2id))]
    with open(path, 'r') as reader:
        raw_data = json.load(reader)
        for domain_n, domain in raw_data.items():
            # Notice: the batch here means few shot batch, not training batch
            batch = domain[0]
            supports_labels = batch['support']['seq_outs']
            all_support_labels = set(collections._chain.from_iterable(supports_labels))
            for lb_from in all_support_labels:
                for lb_to in all_support_labels:
                    if opt.do_debug:  # when debuging, only part of labels are leveraged
                        if lb_from not in label2id or lb_to not in label2id:
                            continue
                    label_mask[label2id[lb_from]][label2id[lb_to]] = 1
    return torch.LongTensor(label_mask)


class MyTokenizer(object):
    def __init__(self, word2id, id2word):
        self.word2id = word2id
        self.id2word = id2word
        self.vocab = word2id

    def convert_tokens_to_ids(self, tokens):
        return [self.word2id.get(token, self.word2id['[PAD]']) for token in tokens]
