import sys
from tqdm import tqdm
from argparse import ArgumentParser
import yaml
from collections import defaultdict
from itertools import permutations
from wsacrebleu.evaluate import Evaluator
from argparse import Namespace
import pandas as pd
import os


def read_config(path):
    with open(path) as config:
        contents = config.read()
        data = yaml.load(contents)
        return data

def get_corpus_langs(data):
    corpus_list = defaultdict(list)
    for corpus in data['corpora']:
        corpus_list[corpus] = data['corpora'][corpus]['langs']
    return corpus_list


def generate_grid(out_dir, corpus, langs):
    data = defaultdict(float)
    langs = sorted(langs)
    df = pd.DataFrame(data, index=langs)
    perm = permutations(langs, 2)
    fpath = os.path.join(out_dir, corpus) 
    for direction in list(perm):
        src, tgt = direction
        args = Namespace(hypothesis='{}/{}-{}.hyp'.format(fpath, src, tgt) \
                       ,references=['{}/{}-{}.ref'.format(fpath, src, tgt)], lang=tgt)
        try:
            evaluator = Evaluator.build(args)
            stats = evaluator.run()
            for key, val in stats.items():
                df.at[src, tgt] = float(val[7:12])
        except:
            df.at[src, tgt] = 0
            pass
    df = df.sort_index(axis=1)                    
    df.to_csv('{}/grid.csv'.format(fpath))

if __name__ == '__main__':
    parser=ArgumentParser()
    parser.add_argument('--out_dir', help='output dir', required=True)
    parser.add_argument('--test_config', help = 'config file used for test', required=True)
    args = parser.parse_args()
    out_dir = args.out_dir
    data = read_config(args.test_config)
    corpora = get_corpus_langs(data)
    
    for corpus in corpora:
        langs = corpora[corpus]
        generate_grid(out_dir, corpus, langs)

    
    