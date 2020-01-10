# PARENTing-rl

Code for PARENTing via Model-Agnostic Reinforcement Learning to Correct Pathological Behaviors in Data-to-Text Generation (Rebuffel, Soulier, Scoutheeten, Gallirani; ACL2020); most of this code is based on [OpenNMT](https://github.com/OpenNMT/OpenNMT-py).

You will need a recent python (3.6+) to use it as is, especially OpenNMT.

Beyond standard packages included in miniconda, usefull packages are torch==1.1.0 torchtext==0.4 and some others required to make onmt work (PyYAML and configargparse for example).

In the following, only instructions for WikiBIO are given. However, most of the time, changing "wikibio" to "webnlg" in the given command will work. Where instructions differ, I'll give both commands.

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

Before any code run, we build an experiment folder to keep things contained

`python create-experiment.py --name wiki-exp`

At this stage, your repository should look like this:

```
.
├── onmt/		             	# Most of the heavy-lifting is done by onmt
├── experiments/ 	           	# Experiments are stored here
│	└── wikibio-exp/
│	│	├── data/
|	│	├── gens/
│	│	└── models/
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
│	└── wikibio-exp
│	│	├── data
│	│	│	├── data.train.0.pt
│	│	│	├── data.valid.0.pt
│	│	│	├── data.vocab.pt
│	│	│	├── preprocess-log.txt
├	│	├── gens
│	│	└── models
├── data						# Dataset is here
└── ...
```

# Training

To train a model within the PARENTing framework, you first need to pretain the model:

`python train.py --config pretrain_wikibio.cfg`

To PARENT a model within our RL framework, you can run:

`python train.py --config train_wikibio.cfg`

To (pre)train with different parameters than the one used in the paper, please refer to my comments in the config file, or check OpenNMT [train doc](http://opennmt.net/OpenNMT-py/options/train.html).

This config files run the training for 100 000 steps, however we manually stop the training before, depending on performance on development set.

# Translating

You can simply translate the test input by running:

`python translate.py --config translate.cfg`

# Evaluation [WIP]

