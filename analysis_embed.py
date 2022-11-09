import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
import argparse
from utils_test import wrapper



def plot(embeds,labels,name,lg1,lg2):
    emb_num = embeds.shape[0]
    plt.clf()
    label_num = 2
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(embeds)
    cmap = cm.get_cmap('tab20')
    plt.title(f'{lg1} and {lg2}-{name}')
    for i,lab in enumerate([lg1,lg2]):
        indices = labels==lab
        plt.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(i*4)).reshape(1,4), label = lab ,alpha=0.5)
    
    s = embeds.shape[0] // 2
    for n,c in zip([20, 110, 130],['r','k','blueviolet']):
        plt.scatter(tsne_proj[[n,n+s],0],tsne_proj[[n,n+s],1], c=c, s=100)

    plt.legend(fontsize='large', markerscale=2)
    plt.savefig(name)
    print('finish plotting '+name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str, 
            choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 
            default='cls', 
            help="Which pooler to use")
    parser.add_argument("--mode", type=str, 
            choices=['dev', 'test', 'fasttest'],
            default='test', 
            help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--language", type=str, 
            default='fr_sw')
    args = parser.parse_args()
    model = wrapper(args)
    lg1 = args.language.split('_')[0]
    lg2 = args.language.split('_')[1]
    texts1 = [line.strip().split(',')[0] for line in open(f'data/multinli_train_{lg1}.csv').readlines()[1:200]]
    emb1 = model.encode_texts(texts1).numpy()
    labels1 = [lg1 for _ in range(len(emb1))]
    texts2 = [line.strip().split(',')[0] for line in open(f'data/multinli_train_{lg2}.csv').readlines()[1:200]]
    emb2 = model.encode_texts(texts2).numpy()
    labels2 = [lg2 for _ in range(len(emb2))]

    embeds = np.concatenate([emb1,emb2],axis=0)
    labels = np.array(labels1 + labels2)
    p_embeds = []
    plot(embeds,labels,args.model_name_or_path.split('/')[-1]+f'_{lg1}_{lg2}'+'.png',lg1,lg2)

if __name__ == '__main__':
    main()