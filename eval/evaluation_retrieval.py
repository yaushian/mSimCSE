import numpy as np
import argparse
from utils_test import wrapper
from sklearn.metrics.pairwise import cosine_similarity as cos
from utils_retrieve import *


def retrieval_eval(emb1, emb2):
    sim = cos(emb1, emb2)
    pred = np.argmax(sim,axis=-1)
    acc = []
    for i,p in enumerate(pred):
        acc.append(i==p)
    return np.mean(acc),pred


def tatoeba(args,model):
    data_dir = 'data/tatoeba/v1/'
    all_acc1,all_acc2 = [],[]
    lang3_dict = {'ara':'ar', 'heb':'he', 'vie':'vi', 'ind':'id',
    'jav':'jv', 'tgl':'tl', 'eus':'eu', 'mal':'ml', 'tam':'ta',
    'tel':'te', 'afr':'af', 'nld':'nl', 'eng':'en', 'deu':'de',
    'ell':'el', 'ben':'bn', 'hin':'hi', 'mar':'mr', 'urd':'ur',
    'tam':'ta', 'fra':'fr', 'ita':'it', 'por':'pt', 'spa':'es',
    'bul':'bg', 'rus':'ru', 'jpn':'ja', 'kat':'ka', 'kor':'ko',
    'tha':'th', 'swh':'sw', 'cmn':'zh', 'kaz':'kk', 'tur':'tr',
    'est':'et', 'fin':'fi', 'hun':'hu', 'pes':'fa', 'aze': 'az',
    'lit': 'lt','pol': 'pl', 'ukr': 'uk', 'ron': 'ro'}
    lang2_dict = {l2: l3 for l3, l2 in lang3_dict.items()}
    lg_list = 'ar he vi id jv tl eu ml ta te af nl de el bn hi mr ur fa fr it pt es bg ru ja ka ko th sw zh kk tr et fi hu az lt pl uk ro'.split()
    lg_list14 = 'ar bg zh de el fr hi ru es sw th tr ur vi'.split()
    lg_list36 = 'af bn et eu fi he hu id it jv ja ka kk ko ml mr nl fa pt ta te tl'.split()
    
    for lg2 in lg_list14:
        #print(f'evaluate {lg}:')
        lg = lang2_dict[lg2]
        f1 = f'{data_dir}tatoeba.{lg}-eng.{lg}'
        text1 = [line.strip() for line in open(f1)]
        emb1 = model.encode_texts(text1)

        f2 = f'{data_dir}tatoeba.{lg}-eng.eng'
        text2 = [line.strip() for line in open(f2)]
        emb2 = model.encode_texts(text2)

        acc1,pred1 = retrieval_eval(emb1.numpy(),emb2.numpy())

        """
        out_file = 'xtreme/my_test_data/predictions/tatoeba/' + f'test-{lg2}.tsv'
        out_file2 = 'xtreme/my_test_data/pairs/tatoeba/' + f'pair-{lg2}.tsv'
        fout = open(out_file,'w')
        fout2 = open(out_file2, 'w')
        for i,p in enumerate(pred1):
            fout.write(str(p) + '\n')
            fout2.write(text1[i]+'\t'+ text2[p] +'\t' + text2[i] + '\n')
        """

        acc2,pred2 = retrieval_eval(emb2.numpy(),emb1.numpy())
        all_acc1.append(acc1)
        all_acc2.append(acc2)
        print(f'{lg} average acc:',(acc1+acc2)/2.)
    
    print('average accuracy 14 to eng:',np.mean(all_acc1))
    print('average accuracy 14 to src:',np.mean(all_acc2))
    for lg2 in lg_list36:
        #print(f'evaluate {lg}:')
        lg = lang2_dict[lg2]
        f1 = f'{data_dir}tatoeba.{lg}-eng.{lg}'
        text1 = [line.strip() for line in open(f1)]
        emb1 = model.encode_texts(text1)

        f2 = f'{data_dir}tatoeba.{lg}-eng.eng'
        text2 = [line.strip() for line in open(f2)]
        emb2 = model.encode_texts(text2)

        acc1,pred1 = retrieval_eval(emb1.numpy(),emb2.numpy())

        """
        out_file = 'xtreme/my_test_data/predictions/tatoeba/' + f'test-{lg2}.tsv'
        out_file2 = 'xtreme/my_test_data/pairs/tatoeba/' + f'pair-{lg2}.tsv'
        fout = open(out_file,'w')
        fout2 = open(out_file2, 'w')
        for i,p in enumerate(pred1):
            fout.write(str(p) + '\n')
            fout2.write(text1[i]+'\t'+ text2[p] +'\t' + text2[i] + '\n')
        """
        acc2,pred2 = retrieval_eval(emb2.numpy(),emb1.numpy())
        all_acc1.append(acc1)
        all_acc2.append(acc2)
        print(f'{lg} average acc:',(acc1+acc2)/2.) 
    print('average accuracy 36 to eng:',np.mean(all_acc1))
    print('average accuracy 36 to src:',np.mean(all_acc2))
    
    acc_ana = []
    for lg in ['hin','fra','deu','afr','tel','tgl','gle','kat','amh','swh']:
        #print(f'evaluate {lg}:')
        f1 = f'{data_dir}tatoeba.{lg}-eng.{lg}'
        text1 = [line.strip() for line in open(f1)]
        emb1 = model.encode_texts(text1)

        f2 = f'{data_dir}tatoeba.{lg}-eng.eng'
        text2 = [line.strip() for line in open(f2)]
        emb2 = model.encode_texts(text2)

        acc1,pred1 = retrieval_eval(emb1.numpy(),emb2.numpy())
        acc2,pred2 = retrieval_eval(emb2.numpy(),emb1.numpy())
        acc_ana.append( '$'+str( round((acc1+acc2)*100.0/2.0,1) )+'$' )
    print(' & '.join(acc_ana))


def bucc_f1(labels, predictions, language=None):
  """Calculate F1 score for BUCC data."""
  labels = set([tuple(l.split('\t')) for l in labels])
  predictions = set([tuple(l.split('\t')) for l in predictions])
  ncorrect = len(labels.intersection(predictions))
  if ncorrect > 0:
    prec = ncorrect / len(predictions)
    rec = ncorrect / len(labels)
    f1_val = 2 * prec * rec / (prec + rec)
  else:
    prec = rec = f1_val = 0
  return {'f1': f1_val * 100, 'precision': prec * 100, 'recall': rec * 100}


def bucc(args, model):
    def proc_text(name):
        all_text,all_idx,id2text = [],[],{}
        fout = open(name + '.txt','w')
        for line in open(name):
            idx, text = line.strip().split('\t')
            all_idx.append(idx.strip())
            all_text.append(text.strip())
            id2text[idx] = text.strip()
            fout.write(idx.strip()+'\n')
        fout.close()
        return all_idx,all_text,id2text

    data_dir = 'data/bucc2018/'
    p2id = []
    all_f1 = []
    for lg in ['fr','ru','zh','de']:
        src_file = data_dir + f'{lg}-en.dev.{lg}'
        tgt_file = data_dir + f'{lg}-en.dev.en'
        gt_file = data_dir + f'{lg}-en.dev.gold'

        """
        out_file = 'xtreme/my_test_data/predictions/bucc2018/' + f'test-{lg}.tsv'
        fout = open(out_file,'w')
        fout2 = open(f'xtreme/my_test_data/pairs/bucc2018/pair-{lg}.txt','w')
        """

        src_id,src_text,src_id2text = proc_text(src_file)
        src_emb = model.encode_texts(src_text).numpy()

        tgt_id,tgt_text,tgt_id2text = proc_text(tgt_file)
        tgt_emb = model.encode_texts(tgt_text).numpy()

        cand2score_file = data_dir + 'temp.tsv'
        mine_bitext(src_emb, tgt_emb, src_file+'.txt', tgt_file+'.txt', cand2score_file)

        candidate2score = {}
        for line in open(cand2score_file):
            line_s = line.strip().split('\t')
            candidate2score[tuple(line_s[1:])] = float(line_s[0]) 

        gt = [line.strip() for line in open(gt_file)]

        threshold = bucc_optimize(candidate2score, gt)
        pred = []
        for cand in candidate2score:
            if candidate2score[cand] >= threshold:
                pred.append('\t'.join(cand))
                #fout.write('\t'.join(cand)+'\n')
                #fout2.write(src_id2text[cand[0]] + '\t' + tgt_id2text[cand[1]] + '\n')
        
        result = bucc_f1(gt, pred)
        all_f1.append(result['f1'])
        print(lg,result)
    print('bucc f1:',np.mean(all_f1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str, 
            choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 
            default='cls_before_pooler', 
            help="Which pooler to use")
    parser.add_argument("--mode", type=str, 
            choices=['dev', 'test', 'fasttest'],
            default='test',
            help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    args = parser.parse_args()
    model = wrapper(args)

    tatoeba(args,model)
    bucc(args,model)


if __name__ == '__main__':
    main()