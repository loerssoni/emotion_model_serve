"""
Adapted from https://github.com/baaesh/CNN-sentence-classification-pytorch/
"""
import re
from torchtext import data
import numpy as np
import pickle
from gensim.models import KeyedVectors, Word2Vec

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", "s", string)
    string = re.sub(r"\'ve", "ve", string)
    string = re.sub(r"n\'t", "nt", string)
    string = re.sub(r"\'re", "re", string)
    string = re.sub(r"\'d", "d", string)
    string = re.sub(r"\'ll", "ll", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
            
def getVectors(args, data):
    """
    

    Parameters
    ----------
    args : argparse object of arguments fed to train
    data : object of class DATA.

    Returns
    -------
   numpy array of word vectors for each text in data

    """
    vectors = []

    if args.mode != 'rand':
        if args.embeddings == 'word2vec':
            embed = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
        if args.embeddings == 'fasttext':
            embed = KeyedVectors.load_word2vec_format('./data/wiki-news-300d-1M.vec', encoding='utf-8')
        if args.embeddings == 'ownfast':
            embed = KeyedVectors.load(f'./data/own_fast_{args.word_dim}.vec', mmap='r')
        else:
            embed = KeyedVectors.load(f'./data/own_vec_{args.word_dim}.vec', mmap='r')
            
        for i in range(len(data.TEXT.vocab)):
            word = data.TEXT.vocab.itos[i]
            if word in embed.vocab:
                vectors.append(embed[word])
            else:
                vectors.append(np.random.uniform(-0.01, 0.01, args.word_dim))
    else:
        for i in range(len(data.TEXT.vocab)):
            vectors.append(np.random.uniform(-0.01, 0.01, args.word_dim))

    return np.array(vectors)

class TextDataset(data.Dataset):
    """
    Wrapper class to enable the sort_key required by bucketiterator
    """
    
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

        
class DATA():
    """
    Class defining the full dataset to be fed into the torch model.
    """
    def __init__(self):
        
        #defield torch dataset field type objects
        self.TEXT = data.Field(batch_first=True, lower=True, fix_length=70)
        with open('data/vocab.pkl', 'rb') as f:
            self.TEXT.vocab = pickle.load(f)