Royi Rassin
Shon Otmazgin

This is a repo for multiNLI_encoder.

This repo replicates the paper [Shortcut-Stacked Sentence Encoders for the MultiNLI inference](https://arxiv.org/pdf/1708.02312.pdf). 
1. Download GloVe pretrained embedding (run script below):
```
mkdir data
cd data
wget https://nlp.stanford.edu/data/glove.6B.zip
wget https://nlp.stanford.edu/data/glove.42B.300d.zip
wget https://nlp.stanford.edu/data/glove.840B.300d.zip

unzip glove.6B.zip
unzip glove.42B.300d.zip
unzip glove.840B.300d.zip
```

2. Install requirements:

Using pip:
```
pip install -r requirements.txt
```

Using conda (you may need to add the appropriate channels):
```
conda create -n snli_env --file conda_requirements.txt
```

3. Re-produce results:

Note: The default embedding file is ```data/glove.8B.300d.txt``` (for faster training) 
      you can change it to larger by modifying ```emb_file``` variable in ```train.py``` (this is straightforward)
```
python train.py
```
