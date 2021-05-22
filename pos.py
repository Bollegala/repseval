"""
Evaluate pretrained word embeddings for part-of-speech tagging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse

import gensim
from gensim.models import KeyedVectors

import pandas as pd
from tabulate import tabulate

from datasets import list_datasets, load_dataset, list_metrics, load_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)

torch.manual_seed(1)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, weight=None):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if weight is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(weight)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long).to(device)

def assign_ix(dataset, word_to_ix):
    L = []
    for x in dataset:
        ltokens = [w.lower() for w in x['tokens']]
        L.append((ltokens, x['pos_tags']))
        for word in ltokens:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return L


def get_embeddings(fname, word_to_ix):
    """
    Load the embeddings for the words in the word_list from the embedding file given by fname.
    Return the data in an nn.embedding layer and an index to the rows. We will assign zero vectors
    for the missing words that are not in the embedding file.
    """
    if fname.endswith('bin'):
        binary = True
    else:
        binary = False
    embedding = KeyedVectors.load_word2vec_format(fname, binary=binary, unicode_errors='ignore')
    T = np.random.rand(len(word_to_ix), embedding.vector_size)
    for word in word_to_ix:
        if word in embedding.key_to_index:
            T[word_to_ix[word],:] = embedding.get_vector(word)  
    return torch.from_numpy(T).to(device)

def check_tags(sent, tags, ix_to_tag):
    txt = ""
    for i in range(len(sent)):
        txt += sent[i] + "/" + ix_to_tag[tags[i]] + " "
    print(txt)


def process(embed_fname, dim):
    # load conll-2003 dataset and compute the word and tag sets.
    conll = load_dataset('conll2003')
    tag_to_ix = {}
    pos_tags = ["\"", "''", "#", "$", "(", ")", ",", ".", ":", "``", "CC", "CD", "DT", 
                "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", 
                "NNS", "NN|SYM", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP",
                "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]
    for tag in pos_tags:
        tag_to_ix[tag] = len(tag_to_ix)
    ix_to_tag = {value:key for key, value in tag_to_ix.items()}

    word_to_ix = {}    

    train_data = assign_ix(conll['train'], word_to_ix)
    test_data = assign_ix(conll['test'], word_to_ix)

    #check_tags(train_data[0][0], train_data[0][1], ix_to_tag)

    # train a model and save to disk.
    model_fname = "pos.model"

    model = train_model(train_data, dim, embed_fname, word_to_ix, tag_to_ix)    
    torch.save(model, model_fname)

    #model = torch.load(model_fname)
    return test_model(model, test_data, word_to_ix, tag_to_ix, ix_to_tag)
    

def train_model(train_data, dim, embed_fname, word_to_ix, tag_to_ix):            
    EMBEDDING_DIM = dim
    HIDDEN_DIM = 100
    NO_EPOCHS = 10

    print("Vocab size = ", len(word_to_ix))
    print("Total tags = ", len(tag_to_ix))
    print("No of epochs =", NO_EPOCHS)
    print("Embedding dim =", EMBEDDING_DIM)

    pwe = get_embeddings(embed_fname, word_to_ix)

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), weight=pwe).double().to(device)
    #print(next(model.parameters()).is_cuda) # True if the model is on GPU

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    pbar = tqdm(desc="Training", total=NO_EPOCHS * len(train_data))
    for epoch in range(NO_EPOCHS):
        for sentence, tags in train_data:
            #print(sentence, tags)
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            #targets = prepare_sequence(tags, tag_to_ix)
            targets = torch.tensor(tags, dtype=torch.long).to(device)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            pbar.update(1)
    pbar.close()
    return model

def test_model(model, test_data, word_to_ix, tag_to_ix, ix_to_tag):
    pbar = tqdm(desc="Testing", total=len(test_data))
    corrects = {} # hold the correctly predicted count for each tag
    total_correct = total_tokens = 0
    pred_count = {} # number of times each tag was predicted
    tag_count = {} # numbef of times each tag in test data

    df = pd.DataFrame()
    for tag in tag_to_ix:
        corrects[tag], pred_count[tag], tag_count[tag] = 0, 0, 0

    with torch.no_grad():
        for sentence, tags in test_data:
            inputs = prepare_sequence(sentence, word_to_ix)
            tag_scores = model(inputs)
            pred_tags = [ix_to_tag[x.item()] for x in torch.argmax(tag_scores, dim=1)]
            pbar.update(1)
            #print(tags, pred_tags)
            targets = [ix_to_tag[x] for x in tags]
            for i in range(len(targets)):
                pred_count[pred_tags[i]] += 1
                tag_count[targets[i]] += 1
                total_tokens += 1
                if pred_tags[i] == targets[i]:
                    corrects[pred_tags[i]] += 1
                    total_correct += 1
    pbar.close()
    # Compute Precision, Recall, F for each tag
    acc = float(100 * total_correct) / float(total_tokens)
    print("Accuracy = %f (%d / %d)" % (acc, total_correct, total_tokens))
    macro_precision = macros_recall = macro_F = 0
    for tag in tag_to_ix:
        precision = 0 if pred_count[tag] == 0 else float(100 * corrects[tag]) / pred_count[tag]
        recall = 0 if tag_count[tag] == 0 else float(100 * corrects[tag]) / tag_count[tag]
        F = 0 if (precision * recall == 0) else (2 * precision * recall) / (precision + recall)
        df = df.append(pd.DataFrame({'precision':precision, 'recall':recall, 'F':F}, index=[tag]))
    macro_precision = df['precision'].mean()
    macro_recall = df['recall'].mean()
    macro_F = df['F'].mean()
    df = df.append(pd.DataFrame({'precision':macro_precision, 'recall':macro_recall, 'F':macro_F}, index=["Macro"]))
    print(tabulate(df, headers='keys', tablefmt='psql'))
    return {'pos_precision':macro_precision, 'pos_recall':macro_recall, 'pos_F':macro_F, 'pos_accuracy':acc}

def guess_dim(fname):
    """
    This funcion guesses the dimensionality of the embedding.
    There can be two types of files: with a header (vocabsize and dim separated by a space)
    and without a header (the first line is the first embedding vector).
    """
    with open(fname) as F:
        first_line = F.readline().strip()
        p = first_line.split()
        if len(p) == 2:
            # this is the header line
            return int(p[1])
        else:
            # this is the first embedding vector
            return len(p) - 1 #first element is the word itself. 

def cli():
    #"../../Meta-Embedding-Framework/data/common_vocab/glove.840B.300d.txt.selected"
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="embedding file")
    args = parser.parse_args()
    process(args.f, guess_dim(args.f))
    pass      

def conv2gensim():
    """
    Sometimes the word embedding file might not have a header indicating vocabulary size
    and the dimensionality. This code add such a header to the embedding file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="input file")
    parser.add_argument("-o", type=str, help="output file")
    args = parser.parse_args()
    # guess dim and vocab size
    dim = vocab_size = 0
    with open(args.i) as in_file:
        first_line = True
        for line in in_file:
            if first_line:
                first_line = False
                dim = len(line.split()) - 1
            vocab_size += 1

    with open(args.o, 'w') as out_file:
        out_file.write("%d\t%d\n" % (vocab_size, dim))
        with open(args.i) as in_file:
            for line in in_file:
                out_file.write("%s" % line)
    pass

if __name__ == "__main__":
    cli()
    #conv2gensim()
    #process(embed_fname, dim)