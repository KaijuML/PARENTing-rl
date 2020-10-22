# PARENTing-rl

Code for [PARENTing via Model-Agnostic Reinforcement Learning to Correct Pathological Behaviors in Data-to-Text Generation (Rebuffel, Soulier, Scoutheeten, Gallinari; INLG2020)](https://arxiv.org/abs/2010.10866); most of this code is based on [OpenNMT](https://github.com/OpenNMT/OpenNMT-py).

You will need a recent python (3.6+) to use it as is, especially OpenNMT.

Beyond standard packages included in miniconda, usefull packages are torch==1.1.0 torchtext==0.4 and some others required to make onmt work (PyYAML and configargparse for example).

In the following, only instructions for WikiBIO are given. However, most of the time, changing "wikibio" to "webnlg" in the given command will work. Where instructions differ, I'll give both commands. In the paper we report experiments on several model. Here we give instructions for LSTM+RL. Modifying instructions to work with others models is intuitive.

# Datasets

There are two datasets used in the paper. Follow `README.md` in data for download and formating instructions.
Once datasets are downloaded and formated, your repository should look like this:

```
.
├── onmt/                   	    # Most of the heavy-lifting is done by onmt
├── data/   					    # Dataset is here
│   ├── wikibio/				    # WikiBIO dataset is stored here
│   │   ├── raw/				    # Raw dataset is stored here
│   │   ├── processed_data/			# Intermediate files are stored here
│   │   ├── train_input.txt			# final txt files
│   │   ├── train_output.txt		# final txt files
│   |   └── ...
│   ├── make-dataset.py			    # next three are formating scripts
│   ├── format_wikibio.py
│   ├── format_webnlg.py
│   └── ...
└── ...
```

# Experiments

Before any code run, we build experiment folders to keep things contained

```
python create-experiment.py --dataset wikibio --name pretraining-lstm
python create-experiment.py --dataset wikibio --name lstm-rl
```

At this stage, your repository should look like this:

```
.
├── onmt/		             	# Most of the heavy-lifting is done by onmt
├── experiments/ 	           	# Experiments are stored here
│	└── wikibio/
│	│   └── pretraining-lstm/
│	│	│	├── data/
|	│	│	├── gens/
│	│	│	└── models/
├── data/						# Dataset is here
└── ...
```

# Preprocessing

Before training models via OpenNMT, you must preprocess the data. I've handled all useful parameters with config files. Please check it out if you want to tweak things, I have tried to include comments on each command. For futher info you can always check out the OpenNMT [preprocessing doc](http://opennmt.net/OpenNMT-py/options/preprocess.html)

`python preprocess.py --config preprocess_wikibio.cfg`

At this stage, your repository should look like this:

```
├── onmt		             	# Most of the heavy-lifting is done by onmt
├── experiments 	           	# Experiments are stored here
│	└── wikibio/
│	│   └── pretraining-lstm/
│	│	│	├── data/
│	│	│	│	├── data.train.0.pt
│	│	│	│	├── data.valid.0.pt
│	│	│	│	├── data.vocab.pt
│	│	│	│	├── preprocess-log.txt
├	│	│	├── gens/
│	│	│	└── models/
├── data						# Dataset is here
└── ...
```

# Training

To train a model within the PARENTing framework, you first need to pretain the model:

`python train.py --config pretrain_wikibio_lstm.cfg`

To PARENT a model within our RL framework, you can run:

`python train.py --config train_wikibio_lstm_rl.cfg`

To (pre)train with different parameters than the one used in the paper, please refer to my comments in the config file, or check OpenNMT [train doc](http://opennmt.net/OpenNMT-py/options/train.html).

In particular, when further training a model via RL, you can select the best checkpoint from pretraining with `--train_from <path_to_cp>`. You can also experiment with a different weighting of MLE / RL losses, using `--rl_gamma_loss` (see paper for details).

This config files run the training for 100 000 steps, however we manually stop the training before, depending on performance on development set.

(Please note that all pretraining/training config files refer to data preprocessed by onmt and placed in the wikibio/pretraining_lstm experiment. This is to reduce redunduncies because the preprocessing step is the same for all models.)

# Translating

You can simply translate the test input by running:

`python translate.py --config translate.cfg`

If you wish to make multiple translate in a row (for exemple to find the best performing checkpoint) you can you the `batch_translate.py` helper:

`python batch_translate --config translate.cfg --dataset wikibio --setname valid --bsz 64 --bms 10`

# Evaluation

To compute PARENT scores,  you can follow  `parent/README.md` for instructions on how to compute PARENT scores, either in command line or in a notebook for better visualization. Note that while working on this project, I used the `parent.py` file at the root of this repo. Since then, I have released a stand alone PARENT repository, which I have included in this one.

You can evaluate the BLEU score using [SacreBLEU](https://github.com/mjpost/sacreBLEU) from [Post, 2018](aclweb.org/anthology/W18-6319). See the repo for installation, it should be a breeze with pip.

You can get the BLEU score by running:

`cat experiments/wikibio/pretraining-lstm/gens/test/predictions.txt | sacrebleu --force data/wikibio/test_output.txt`

(Note that --force is not required as it doesn't change the score computation, it just suppresses a warning because this dataset is always tokenized which is not good practice in general due to different tokenization habits from different researchers, but we have no choice here because wikibio ships already tokenized.)

Alternatively you can use any prefered method for BLEU computation. I have also checked scoring models with [NLTK](aclweb.org/anthology/W18-6319) and scores were virtually the same.

An example script to compute PARENT and BLEU (using NLTK):

```python
from nltk.translate.bleu_score import corpus_bleu
from parent.parent import parent as corpus_parent
import numpy as np
import json, os

with open('data/wikibio/test_tables.jl', mode="r", encoding='utf8') as f:
    tables = [json.loads(line) for line in f if line.strip()]

with open('data/wikibio/test_output.txt', mode="r", encoding='utf8') as f:
    references = [line.strip().split() for line in f if line.strip()]

len(tables) == len(references)

experiment_folder = 'experiments/wikibio/pretraining-lstm/gens/test/'

res = dict()
for filename in os.listdir(experiment_folder):

    with open(os.path.join(experiment_folder, filename), mode="r", encoding='utf8') as f:
        predictions = [line.strip().split() for line in f if line.strip()]

    assert len(tables) == len(references) == len(predictions)
    
    p, r, f = corpus_parent(predictions, references, tables)
    b = corpus_bleu([[r] for r in references], predictions)
    res[filename] = {
        'parent': (p, r, f),
        'bleu': b
    }
    
with open("scores.json") as f:
    json.dump(res, f)
    
for name, scores in res.items():
    print(name)
    print(f"\tPARENT (Prec, Rec, F1): {', '.join(map(str, [np.round(100*s, 2) for s in scores['parent']]))}")
    print(f"\tBLEU - - - - - - - - -: {np.round(100*scores['bleu'], 2)}")
    print()
```
