import os
import random
import json

raw_dir = '/mnt/sda/f/shdata/snips/'
new_dir = '/mnt/sda/f/shdata/zero-shot-dataset/'


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
        slots = []
        vals = []
        for i in range(len(seq_in)):
            temp_slots = []
            temp_vals = []
            temp_slots = list(slots2val.keys())
            for j in temp_slots:
                slots_val = random.sample(slots2val[j], 5)
                temp_vals.append(slots_val)
            slots.append(temp_slots)
            vals.append(temp_vals)
        domain[intent]['slots'] = slots
        domain[intent]['slot_vals'] = vals
    return domain

def div_7_src_dev_tgt(domain: dict):
    tgts = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']
    devs = ['SearchCreativeWork', 'SearchScreeningEvent', 'PlayMusic', 'RateBook', 'GetWeather', 'BookRestaurant','AddToPlaylist']
    srcs = []
    intents = list(domain.keys())
    for i in range(len(tgts)):
        tmp = []
        for inte in intents:
            if inte != tgts[i] and inte != devs[i]:
                tmp.append(inte)
        srcs.append(tmp)
    for i in range(1, 8):
        trainpath = new_dir + 'snips_train_%d.json' % (i)
        devpath = new_dir + 'snips_valid_%d.json' % (i)
        testpath = new_dir + 'snips_test_%d.json' % (i)



        with open(devpath, 'w') as f:
            d = json.dumps(domain[devs[i-1]])
            f.write(d)
        f.close()

        with open(testpath, 'w') as f:
            d = json.dumps(domain[tgts[i-1]])
            f.write(d)
        f.close()

        with open(trainpath, 'w') as f:
            src_dict = {'seq_in':[], 'seq_out':[], 'slots':[], 'slot_vals':[]}

            for j in srcs[i-1]:
                src_dict['seq_in'] += domain[j]['seq_in']
                src_dict['seq_out'] += domain[j]['seq_out']
                src_dict['slots'] += domain[j]['slots']
                src_dict['slot_vals'] += domain[j]['slot_vals']
            d = json.dumps(src_dict)
            f.write(d)
        f.close()







# print(domain['PlayMusic']['seq_out'][0])
domain = process_data_by_intent()        
domain = paired_slot_name_and_val(domain)
div_7_src_dev_tgt(domain)
# print(domain['AddToPlaylist']['slot_vals'])
