import os
import random

raw_dir = '/mnt/sda/f/shdata/snips/'

def process_data_by_intent():
    domain = {}
    for lst in ['train', 'valid', 'test']:
        raw_path = raw_dir + lst
        seq_in = []
        seq_out = []
        label = []
        for line in open(raw_path + '/seq.in', 'r'):
            seq_in.append(line.strip())
        for line in open(raw_path + '/seq.out', 'r'):
            seq_out.append(line.strip())
        for line in open(raw_path + '/label', 'r'):
            label.append(line.strip())    
        assert len(seq_in) == len(seq_out) and len(seq_out) == len(label)
        for i in range(len(label)):
            if(label[i] not in domain):
                domain[label[i]] = {'seq_in':[seq_in[i].split(' ')],'seq_out':[seq_out[i].split(' ')]}
            else:
                domain[label[i]]['seq_in'].append(seq_in[i].split(' '))
                domain[label[i]]['seq_out'].append(seq_out[i].split(' '))
    return domain

domain = process_data_by_intent()
# print(domain['PlayMusic']['seq_out'][0])

def paired_slot_name_and_val(domain: dict):

    for intent in domain.keys():
        slots2val = {}
        seq_in = domain[intent]['seq_in']
        seq_out = domain[intent]['seq_out']
        for utterance, labels in zip(seq_in, seq_out):
            for j in range(len(labels)):
                if labels[j][0] == 'B':
                    slot = labels[j][2:]
                    if slot not in slots2val:
                        slots2val[slot] = []
                    k = j+1
                    while k < len(labels) and labels[k][0] == 'I':
                        k += 1
                    k -= 1
                    temp_slot = []
                    for w in range(j, k+1):
                        temp_slot.append(utterance[w])
                    slots2val[slot].append(temp_slot)
        
        
paired_slot_name_and_val(domain)