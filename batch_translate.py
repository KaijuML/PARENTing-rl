import subprocess
import functools
import argparse
import os, re


partial_shell = functools.partial(subprocess.run, shell=True,
                                  stdout=subprocess.PIPE)
def shell(cmd):
    """Execute cmd as if from the command line"""
    completed_process = partial_shell(cmd)
    return completed_process.stdout.decode("utf8")

def posint(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive int")
    return ivalue

def strposint(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a strictly positive int")
    return ivalue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', dest='dataset', default='wikibio',
                        choices=['wikibio', 'webnlg'])
    parser.add_argument('--setname', dest='setname', default='test',
                        choices=['test', 'dev'])
    parser.add_argument('--experiment', '-e', dest='experiment', 
                        default='pretraining-sarnn')
    parser.add_argument('--start-step', dest='start_step', default=0, type=posint)
    parser.add_argument('--step-size', dest='step_size', default=1, type=strposint)
    parser.add_argument('--bms', dest='bms', default=1, type=strposint,
                        help="beam size")
    parser.add_argument('--bsz', dest='bsz', default=64, type=strposint,
                        help="batch size")
    parser.add_argument('--blk', dest='blk', default=0, type=posint, 
                        help="block ngram repeats")
    parser.add_argument('--gpu', dest='gpu', default=0, type=posint)
    
    args = parser.parse_args()
    
    print(f"Batch translating models from experiment {args.experiment}")
    
    exp_dir = os.path.join('experiments', args.dataset, args.experiment)
    mdl_dir = os.path.join(exp_dir, 'models')
    gns_dir = os.path.join(exp_dir, 'gens', args.setname)
    
    def get_step(fname):
        return int("".join(re.findall("([0-9]+?)[.]pt", fname)))
    
    models = [fname for fname in os.listdir(mdl_dir)]
    models = sorted(models, key=get_step, reverse=False)

    src = os.path.join('data', args.dataset, f'{args.setname}_input.txt')
    
    n_processed = -1
    for idx, fname in enumerate(models):
        n_processed += 1
        if n_processed % args.step_size:
            print(f"Skipping step {step}")
            continue
            
        print(idx, "translating", fname)
        
        model = os.path.join(mdl_dir, fname)
        output_pfx = f'bms{args.bms}.blk{args.blk}.bsz{args.bsz}-step_{get_step(fname)}'
        output = os.path.join(gns_dir, f'{output_pfx}.txt')
        log_file = os.path.join(exp_dir, 'translate-log.txt')
        
        cmd_args = [
            f'-model {model}',
            f'-src {src}',
            f'-output {output}',
            f'-beam_size {args.bms}',
            f'-block_ngram_repeat {args.blk}',
            f'-batch_size {args.bsz}',
            f'-gpu {args.gpu}',
            f'-log_file {log_file}'
        ]
        
        cmd_args = ' '.join(cmd_args)
        cmd = f'python translate.py --config translate.cfg {cmd_args}'
        print(cmd)
        
        _ = shell(cmd)
