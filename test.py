"""
Adapted from https://github.com/baaesh/CNN-sentence-classification-pytorch/
"""


import argparse
import json

import torch
from torch import nn

from model import CNNSentence
from cnndata import DATA, getVectors

from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score


def test(model, data, mode='test'):
    """
    Runs testing for a given model and a given dataset
    Parameters
    ----------
    model : TYPE
        A trained torch classification model.
    data : object of class DATA
    mode : str
        Defines, whether to run the test on dev set or test set. The default is 'test'.

    Returns
    -------
    loss : loss for the dataset
    acc : accuracy of model
    preds: torch tensor of shape [n, 2] with predictions and labels

    """
    #create iterator from input dataset
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)
    
    #define loss and prep model for evaluation
    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0
    preds = torch.empty(0, 2)
    #run test for batch iterators
    for batch in iterator:
        pred = model(batch)
        #evaluate loss for batch
        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.item()
        #predict and compare to labels to get sum of correct predictions
        _, pred = pred.max(dim=1)
        pred_label = torch.stack((pred, batch.label), 0)
        pred_label = torch.transpose(pred_label, 0, 1)
        preds = torch.cat((preds, pred_label), 0)
        acc += (pred == batch.label).sum().float()
        size += len(pred)
    #divide by n to get accuracy
    acc /= size
    acc = acc.cpu().item()
    return loss, acc, preds


def load_args(args):
    """
    Helper function to load model args from file    
    """
    with open(f'saved_models/args{args.model_time}.txt', 'r') as f:
        arg_dict = json.load(f)
    args.__dict__.update(arg_dict)
    return args

def load_model(args, data, vectors):
    """
    Helper function to load model from file    
    """
    model_path = f'saved_models/CNN_sentence_{args.model_time}.pt'
    model = CNNSentence(args, data, vectors)
    model.load_state_dict(torch.load(model_path))

    return model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-time', default='09_02_09', type=str,
                        help='Input a model-time of a saved model in format HH_MM_SS')
    args = parser.parse_args()
    args = load_args(args)
    
    print('loading data...')
    data = DATA(args)
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    print('loading vectors...')
    vectors = getVectors(args, data)
    print('loading model...')
    model = load_model(args, data, vectors)
    #get loss, acc and preds from test util
    loss, acc, preds = test(model, data)
    dev_loss, dev_acc, _ = test(model, data, mode='dev')
    
    #transform from torch tensor to array and compute required metrics
    preds = preds.numpy().astype(int)
    conf = multilabel_confusion_matrix(preds[:,1], preds[:,0])  
    prec = precision_score(preds[:,1], preds[:,0], average='weighted')
    rec = recall_score(preds[:,1], preds[:,0], average='weighted')
    
    #restructure the confusion matrix to list
    conf = [[conf[i,:,:].tolist(),data.LABEL.vocab.itos[i]] for i in range(conf.shape[0])]
    #dump to json
    with open(f'saved_models/test_res{args.model_time}.txt', 'w') as f:
         json.dump({'conf':conf,'acc':acc,'loss':loss, 'prec':prec,'rec':rec,
			'dev_loss':dev_loss, 'dev_acc':dev_acc}, f, indent=2)
    
    print(f'test acc: {acc:.3f}')    
    print(f'test loss: {loss:.3f}')
    print(f'test precision: {prec:.3f}')
    print(f'test recall: {rec:.3f}')
    print('Confusion matrices:')
    print(conf)
    
    