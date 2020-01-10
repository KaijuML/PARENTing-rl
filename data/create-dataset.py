from format_wikibio import main as create_wikibio
from format_webnlg import main as create_webnlg
import argparse
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', dest='dataset', default="webnlg",
                       choices=['wikibio', 'webnlg'])
    
    args = parser.parse_args()
    
    if args.dataset == 'wikibio':
        create_wikibio()
    elif args.dataset == 'webnlg':
        raise NotImplementedError('Everything should work fine but I have not '
                                  'yet fixed the paths inside format_webnlg.py')
        create_webnlg()