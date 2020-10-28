def get_preds():
    import json
    import re
    import torch

    from model import CNNSentence
    from cnndata import DATA, getVectors, clean_str
    import pickle
    
    class Args:
        def __init__(self):
            with open(f'cnn.txt', 'r') as f:
                arg_dict = json.load(f)
            self.__dict__ = arg_dict

    def load_model(args, data, vectors):
        """
        Helper function to load model from file    
        """
        model_path = f'cnn.pt'
        model = CNNSentence(args, data, vectors)
        model.load_state_dict(torch.load(model_path))
        return model
    def predict(model, data, sents):
        indexed = [[data.TEXT.vocab.stoi[t] for t in sent] for sent in sents]
        indexed = [[indx + [0] * (70 - len(indx))] for indx in indexed]

        tensor = torch.LongTensor(indexed).to('cpu')
        tensor = tensor.squeeze(1)
        pred = model(tensor)
        return pred.max(1)
    
    labels = ["joy","sadness","anger","fear","love","surprise"]
    args = Args()   
    datamod = DATA()  
    data = {'success':False}
    
    params = {'msg':'what on earth is this;how do you know?;i am so sick of this bull crap;this makes me very very sad'}
    if (params != None):
        sents = params.get("msg").split(';')
        sents = [clean_str(sent).split() for sent in sents]

        vectors = getVectors(args, datamod)
        setattr(args, 'word_vocab_size', len(datamod.TEXT.vocab))
        setattr(args, 'class_size', 6)
        model = load_model(args, datamod, vectors)
        model.eval()
        
        preds = predict(model, datamod, sents)
        data["response"] = ';'.join([labels[pred] for pred in preds[1]])
        data["success"] = True 
    return data
    
if __name__ == "__main__":
    print(get_preds())