from nltk.translate.bleu_score import corpus_bleu
from parent import parent as corpus_parent

import numpy as np
import argparse
import json
import os


def load_tables(dataset, setname):
    
    tables_filename = os.path.join("data", dataset, f"{setname}_tables.jl")
    with open(tables_filename, encoding="utf8", mode="r") as tables_file:
        tables = [json.loads(line) for line in tables_file]
        
    return tables

def load_refs(dataset, setname):
    refs_filename =os.path.join("data", dataset, f"{setname}_output.txt")
    with open(refs_filename, encoding="utf8", mode="r") as refs_file:
        refs = [[line.strip().split(" ")]
                for line in refs_file if line.strip()]
        
    return refs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', dest='dataset', default='wikibio',
                        choices=['wikibio', 'webnlg'])
    parser.add_argument('--setname', dest='setname', default='test',
                        choices=['test', 'dev'])
    parser.add_argument('--experiment', '-e', dest='experiment', 
                        default='pretraining-sarnn')
    parser.add_argument('--recompute', dest='recompute', action='store_true',
                        help="Set to true if you want to compute the metric " \
                        "for all .txt files. By default, we skip files where " \
                        "we already computed a score")
    
    args = parser.parse_args()
    folder = os.path.join("experiments", args.dataset, args.experiment, 'gens', args.setname)
    assert os.path.exists(folder)
    
    print("Loading TABLES and REFERENCES")
    list_of_references = load_refs(args.dataset, args.setname)
    tables = load_tables(args.dataset, args.setname)
    print("TABLES and REFERENCES loaded")
    
    res_filename = os.path.join(folder, 'results.json')
    if os.path.exists(res_filename):
        with open(res_filename, encoding='utf8', mode='r') as resfile:
            res_content = json.load(resfile)
    else:
        res_content = dict()
        
    filenames = [os.path.join(folder, fname)
                 for fname in os.listdir(folder)
                 if fname.endswith('.txt')]
    filenames = sorted(filenames)
    
    for fname in filenames: 
        if fname in res_content and not args.recompute:
            print(f'Scores already computed for {fname}')
            continue
            
        print(f'Computing PARENT and BLEU scores for {fname}')
        
        with open(fname, mode='r', encoding='utf8') as f:
            candidates = [line.strip().split() for line in f if line.strip()]
            
        BLEU = corpus_bleu(list_of_references, candidates)
        PARENT = corpus_parent(candidates, list_of_references, tables)
        
        res = {
            'BLEU': BLEU,
            'PARENT-precision': np.mean(PARENT[0]),
            'PARENT-recall': np.mean(PARENT[1]),
            'PARENT-f1': np.mean(PARENT[2])
        }
        
        print(res)
        res_content[fname] = res
        
        # We dump the dict each time to be more robust against interruption
        with open(res_filename, mode="w", encoding="utf8") as resfile:
            json.dump(res_content, resfile)
