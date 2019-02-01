from datetime import datetime; start = datetime.now()
from time import time; start_time = time()
from logging import getLogger, FileHandler, StreamHandler, Formatter, DEBUG, INFO
import gc, os, random, re

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset


# parameters
debug = False
one_epoch = False
validate = True
# model
emb_size = 300
max_features = 120000
maxlen = 72
embedding_trainable = False
n_capsule = 5
capsule_dim = 5
# data
emb_paths = {
    'glove': '../input/embeddings/glove.840B.300d/glove.840B.300d.txt',
    # 'fasttext': '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
    'paragram': '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt',
    # 'google': '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
}
batch_size = 512
batch_size_val = 8192
# training
device = 'cuda:0'
thresholds = np.arange(0.3, 0.501, 0.01)
n_splits = 4
skf_random_state = 10
epochs = 5
lr = 1e-3
epoch_unfreeze = None  # 3
lr2 = 1e-4
seed0 = 1234
seed = 1029
# logger
log_file = 'log.txt'
if os.path.exists(log_file): os.remove(log_file)  # noqa
fh = FileHandler(log_file)
fh.setLevel(DEBUG)
sh = StreamHandler()
sh.setLevel(INFO)
for handler in [fh, sh]:
    formatter = Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
logger = getLogger(__name__)
logger.setLevel(INFO)
logger.addHandler(fh)
logger.addHandler(sh)


class Attention(nn.Module):
    def __init__(self, hidden_size, step_size, bias=True, **kwargs):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.step_size = step_size

        weight = torch.zeros(hidden_size, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(step_size)) if bias else None

    def forward(self, x, mask=None):
        eij = torch.mm(x.contiguous().view(-1, self.hidden_size), self.weight).view(-1, self.step_size)
        if self.bias is not None:
            eij += self.bias
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        if mask is not None:
            a = a * mask
        a = a / torch.sum(a, dim=1, keepdim=True) + 1e-10
        weighted = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted, dim=1)


class RNNAttn(nn.Module):

    def __init__(
        self, vocab_size, embedding_size=300, embedding=None, embedding_trainable=False,
        hidden_size=64, n_layers=1, device='cuda:0'
    ):
        super(RNNAttn, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = 128
        self.n_layers = n_layers
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding, dtype=torch.float32))
            self.embedding.weight.requires_grad = embedding_trainable
        self.dropout_emb = nn.Dropout(0.1)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers=n_layers, bias=True, batch_first=True, dropout=0.,
            bidirectional=True)
        self.gru = nn.GRU(
            hidden_size * 2, hidden_size, num_layers=n_layers, bias=True, batch_first=True, dropout=0.,
            bidirectional=True)
        self.attn_lstm = Attention(hidden_size * 2, maxlen)
        self.attn_gru = Attention(hidden_size * 2, maxlen)
        self.fc = nn.Linear(hidden_size * 2 * 4, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        bs = x.size(0)
        seq = x.size(1)

        x = self.embedding(x)
        x = self.dropout_emb(x.view(bs * seq, self.embedding_size))  # NOTE
        x = x.view(bs, seq, self.embedding_size)
        h_lstm, _ = self.lstm(x)
        h_gru, _ = self.gru(h_lstm)
        h_attn_lstm = self.attn_lstm(h_lstm)
        h_attn_gru = self.attn_gru(h_gru)
        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)
        out = torch.cat([h_attn_lstm, h_attn_gru, avg_pool, max_pool], 1)
        out = self.relu(self.fc(out))
        out = self.dropout(out)
        out = self.out(out)
        return out.squeeze()


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed


def clean_punctuation(x):
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


def clean_number(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


def replace_typical_misspell(text):
    mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
    mispellings, mispellings_re = _get_mispell(mispell_dict)
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)


def preprocess(df):
    df['question_text'] = df['question_text'].apply(lambda text: clean_punctuation(text.lower()))
    df['question_text'] = df['question_text'].apply(lambda text: clean_number(text))
    df['question_text'] = df['question_text'].apply(lambda text: replace_typical_misspell(text))
    return df


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def get_embedding(path, word_index, name='glove'):
    if name.lower() == 'glove':
        embeddings_index = dict(
            get_coefs(*o.split(' ')) for o in open(path, encoding='utf-8', errors='ignore'))
        emb_mean, emb_std = -0.005838499, 0.48782197
        emb_size = 300
    elif name.lower() == 'fasttext':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(path) if len(o) > 100)
        emb_mean, emb_std = -0.0033469985, 0.109855495
        emb_size = 300
    elif name.lower() == 'paragram':
        embeddings_index = dict(
            get_coefs(*o.split(" "))
            for o in open(path, encoding='utf-8', errors='ignore') if len(o) > 100)
        emb_mean, emb_std = -0.0053247833, 0.49346462
        emb_size = 300
    elif name.lower() == 'google':
        loaded = KeyedVectors.load_word2vec_format(path, binary=True)
        embeddings_index = dict([(k, loaded[k]) for k in loaded.vocab.keys()])
        emb_mean, emb_std = -0.003527845, 0.13315111
        emb_size = 300
    else:
        raise NotImplementedError('No embedding: {}'.format(name))
    logger.debug('Created embedding_index: {}'.format(name))

    n_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (n_words + 1, emb_size))  # NOTE
    for word, i in tqdm(word_index.items()):
        if i >= max_features: continue  # noqa
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    del embeddings_index, embedding_vector; gc.collect()  # noqa
    return embedding_matrix


def f1_score_for_thresholds(y_true, proba):
    best_score, best_thresh = 0., 0.
    for thresh in thresholds:
        score = metrics.f1_score(y_true, np.asarray(proba) > thresh)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_score, best_thresh


class AverageMeter(object):
    """
    Computes and stores the average and current value
    cf. https://github.com/pytorch/examples/blob/master/imagenet/main.py#L296
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step(model, inputs, targets, criterion, device):
    inputs, targets = inputs.to(device), targets.to(device)  # noqa
    out = model(inputs)
    loss = criterion(out, targets)
    return out, loss


def train(model, data_loader, criterion, optimizer, device):
    losses = AverageMeter()
    model.train()
    for inputs, targets in data_loader:
        bs = inputs.size(0)
        out, loss = step(model, inputs, targets, criterion, device)
        losses.update(loss.item(), bs)
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
    return losses.avg


def validate(model, data_loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    outputs, trues = [], []
    with torch.no_grad():
        for inputs, targets in data_loader:
            bs = inputs.size(0)
            out, loss = step(model, inputs, targets, criterion, device)
            losses.update(loss.item(), bs)
            out = torch.sigmoid(out)
            outputs += out.detach().cpu().numpy().tolist()
            trues += targets.detach().numpy().tolist()
    f1, thresh = f1_score_for_thresholds(trues, outputs)
    return {'loss': losses.avg, 'f1': f1, 'thresh': thresh, 'output': np.array(outputs).flatten()}


def test(model, data_loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for (inputs,) in data_loader:
            inputs = inputs.to(device)
            out = model(inputs)
            out = torch.sigmoid(out)
            outputs += out.detach().cpu().numpy().tolist()
    return np.array(outputs).flatten()


def main():
    # data
    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')
    # preprocess
    t0 = time()
    df_train = preprocess(df_train)
    df_test = preprocess(df_test)
    X_train = df_train['question_text'].fillna('_na_').values
    X_test = df_test['question_text'].fillna('_na_').values
    logger.info('Preprocessed in {:.2f}s. Time {:.2f}s'.format(time() - t0, time() - start_time))
    # features
    # tokenize
    t0 = time()
    tokenizer = Tokenizer(lower=True, filters='', num_words=max_features)  # NOTE
    tokenizer.fit_on_texts(list(X_train))  # NOTE + list(X_test))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    logger.info('Tokenized in {:.2f}s. Time {:.2f}s'.format(time() - t0, time() - start_time))
    # padding
    t0 = time()
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)
    logger.info('Padding in {:.2f}s. Time {:.2f}s'.format(time() - t0, time() - start_time))
    # target
    y_train = df_train['target'].values

    # embedding
    if debug:
        embedding = None
    else:
        embeddings = []
        for name, path in emb_paths.items():
            embeddings.append(get_embedding(path, tokenizer.word_index, name=name))
        embedding = np.mean(embeddings, axis=0)
        del embeddings
        gc.collect()
    logger.info('Loaded embeddings. Time {:.2f}s'.format(time() - start_time))
    '''
    X_train = np.load('../input/repro/X_train.npy')
    y_train = np.load('../input/repro/y_train.npy')
    X_test = np.load('../input/repro/X_test.npy')
    embedding = np.load('../input/repro/emb_mean_glove_paragram.npy')
    '''

    # data loader
    test_dataset = TensorDataset(torch.from_numpy(X_test.astype('int64')))
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # train
    seed = set_seed(seed0)
    logger.info('Set seed {}'.format(seed))
    proba_train, proba_test = np.zeros(len(X_train)), np.zeros((len(X_test), n_splits))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=skf_random_state)
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        logger.info('Fold {}'.format(fold_i + 1))
        # split
        X_train_fold, y_train_fold = X_train[train_idx].astype('int64'), y_train[train_idx].astype('float32')
        X_val_fold, y_val_fold = X_train[val_idx].astype('int64'), y_train[val_idx].astype('float32')
        train_dataset = TensorDataset(torch.from_numpy(X_train_fold), torch.from_numpy(y_train_fold))
        val_dataset = TensorDataset(torch.from_numpy(X_val_fold), torch.from_numpy(y_val_fold))
        # data loader
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

        set_seed(seed + fold_i)
        # model
        model = RNNAttn(
            max_features, embedding_size=emb_size, embedding=embedding,
            embedding_trainable=embedding_trainable, device=device).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch_i in range(1, 1 + epochs):
            if debug: break
            message = ''
            if epoch_unfreeze is not None and epoch_i == epoch_unfreeze:
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(model.parameters(), lr=lr / 10)
                message = 'Unfreezed. LR {}'.format(lr / 10)
            loss = train(model, train_loader, criterion, optimizer, device)
            if validate:
                validation = validate(model, val_loader, criterion, device)
                logger.info('Fold {} Epoch {}/{} Train/Loss {:.4f} Val/Loss {:.4f} Val/F1 {:.4f} Threshold {:.2f} {}'.format(
                    fold_i + 1, epoch_i, epochs, loss,
                    validation['loss'], validation['f1'], validation['thresh'],
                    message
                    )
                )
            if one_epoch: break

        logger.info('Fold {} Finished training'.format(fold_i + 1))
        if validate:
            proba_train[val_idx] = validation['output']

        proba = test(model, test_loader, device)
        proba_test[:, fold_i] = proba
        logger.info('Fold {} Scored test probas'.format(fold_i + 1))
        if one_epoch: break

    # submit
    f1, threshold = f1_score_for_thresholds(y_train, proba_train)
    logger.info('Train/F1/Best {:.4f} Threshold {:.2f}'.format(f1, threshold))

    y_pred = (proba_test.mean(axis=1) > threshold).astype('int')
    submit = pd.read_csv('../input/sample_submission.csv')
    submit.prediction = y_pred
    submit.to_csv('submission.csv', index=False)

    elapsed = datetime.now() - start
    logger.info('{}h {}m {}s'.format(*str(elapsed).split(':')))


if __name__ == '__main__':
    main()
