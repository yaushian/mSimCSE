import numpy as np
import argparse
from utils_test import wrapper
from sklearn.metrics.pairwise import cosine_similarity as cos
from scipy.stats import spearmanr, pearsonr
import io

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def sts_eval(emb1, emb2):
    sys_scores = []
    for kk in range(emb2.shape[0]):
        s = cosine(emb1[kk], emb2[kk])
        sys_scores.append(s)
    return sys_scores

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

    input_dir = 'data/Sts2017/STS2017.eval.v1.1/'
    gs_dir = 'data/Sts2017/STS2017.gs/'
    for name in ['track1.ar-ar','track2.ar-en','track3.es-es','track4a.es-en','track4b.es-en','track6.tr-en','track5.en-en']:
        print(name)
        f1 = f'STS.input.{name}.txt'
        f2 = f'STS.gs.{name}.txt'
        sent1, sent2 = zip(*[l.split("\t") for l in io.open(input_dir+f1, encoding='utf8').read().splitlines()])
        sent1,sent2 = sent1[:250],sent2[:250]
        gs = np.array([float(x.strip()) for x in open(gs_dir+f2)])

        emb1,emb2 = model.encode_texts(list(sent1)),model.encode_texts(list(sent2))
        sys_scores = sts_eval(emb1,emb2)

        result = spearmanr(sys_scores, gs)
        print(result[0]*100)

if __name__ == '__main__':
    main()