# Dataset creation

Instructions to download and format datasets for this projects.
Most scripts are reused from previous work on these dataset to ensure results are not from better preprocessing.

Specificaly:

    - wikibio formating script comes from [this great repo](https://github.com/tyliupku/wiki2bio/blob/master/preprocess.py)

Scripts assume you are in the `data/` repository, with the following files:

```
.
├── create-dataset.py              # Wrapper for simplicity
├── format-wikibio.py              # self explanatory
├── format-webnlg.py               # self explanatory
```

## WikiBIO

Download and unpack the dataset the dataset:

```
git clone https://github.com/DavidGrangier/wikipedia-biography-dataset.git
cd wikipedia-biography-dataset
cat wikipedia-biography-dataset.z?? > tmp.zip
unzip tmp.zip
rm tmp.zip
mkdir ../wikibio
mv  wikipedia-biography-dataset/ ../wikibio/raw
cd ..
rm -rf wikipedia-biography-dataset
```

create the dataset:

```
python create-dataset.py --dataset wikibio
```
