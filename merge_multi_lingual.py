import pandas as pd
import sys,os
import random
from collections import defaultdict

def get_data(in_name):
    fin = open(in_name)
    data = []
    for i,line in enumerate(fin.readlines()[1:]):
        l_split = line.strip().split('\t')
        p,h,l = l_split
        p = p.replace(',',' ')
        p = p.replace('"','')
        h = h.replace(',',' ')
        h = h.replace('"','')
        data.append((p,h,l))
    return data

def merge(all_data):
    t2cid = {}
    out_data = []
    all_lg = list(all_data.keys())
    for i,d in enumerate(all_data['en']):
        if d[2] == 'contradictory':
            t2cid[d[0]] = i
    error = 0
    for i,d in enumerate(all_data['en']):
        if d[0] in t2cid:
            cid = t2cid[d[0]]
        else:
            cid = random.choice(list(t2cid.values()))
            error += 1
        if d[2] == 'entailment':
            for lg in all_lg:
                slg = random.choice(all_lg)
                slg2 = random.choice(all_lg)
                out_data.append(all_data[lg][i][0]+','+all_data[slg][i][1]+','+all_data[slg2][cid][1])
    print('error:',error)
    return out_data


data_dir = 'data/XNLI-MT-1.0/multinli/'
all_data = {}
for f in os.listdir('data/XNLI-MT-1.0/multinli/'):
    lg = f.split('.')[2]
    all_data[lg] = get_data(data_dir+f)
out_data = merge(all_data)
print(out_data[10])

fin = open('data/nli_for_simcse.csv')
for line in fin.readlines()[1:]:
    out_data.append(line.strip())
print(out_data[-1])

random.shuffle(out_data)
fout = open('data/nli_merged_all.csv','w')
fout.write('sent0,sent1,hard_neg\n')
fout.write('\n'.join(out_data))