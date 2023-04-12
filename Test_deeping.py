# PytorchでTransformerによって対話システムを構築するコードを作ってください。
# このコードは、TransformerのEncoderのみを用いて、対話システムを構築します。

# 1. データの読み込み
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MeCab
import re
import random
import time
import math
import sys
import os
import unicodedata
import string
import codecs
import glob
import pickle
import itertools
import json
import copy
import datetime
import warnings
warnings.filterwarnings('ignore') 

# 2. データの前処理
# 2.1. データの読み込み

# データの読み込み
data = pd.read_csv('data/ChatbotData.csv')  
# データの確認
data.head()  

# 2.2. データの前処理

# データの前処理
def preprocess_sentence(sentence): 
    # 形態素解析
    tagger = MeCab.Tagger()
    tagger.parse('')
    node = tagger.parseToNode(sentence)
    words = []
    while node:
        # 単語を取得
        word = node.surface
        # 単語が空文字列の場合はスキップ
        if word == '':
            node = node.next
            continue
        # 単語をリストに追加
        words.append(word)
        # 次の単語に移動
        node = node.next
    # 単語のリストをスペースで連結して返す
    return ' '.join(words)

# 前処理を実行
data['Q'] = data['Q'].apply(lambda x: preprocess_sentence(x))
data['A'] = data['A'].apply(lambda x: preprocess_sentence(x))
data.head()

# 2.3. データの分割

# データの分割
train_data = data[:int(len(data)*0.8)]
test_data = data[int(len(data)*0.8):]
print('訓練データの数：', len(train_data))
print('テストデータの数：', len(test_data))

# 2.4. 単語の辞書の作成

# 単語の辞書の作成
def create_vocab(data):
    # 単語の辞書を作成
    vocab = {}
    for sentence in data['Q'].values.tolist() + data['A'].values.tolist():
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
    # 単語の辞書を作成
    vocab = {word: i for i, (word, freq) in enumerate(vocab.items(), 1)}
    return vocab

# 単語の辞書を作成
vocab = create_vocab(data)
print('単語の数：', len(vocab))

# 2.5. 単語の辞書の保存

# 単語の辞書の保存
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)


# 3. データのローダーの作成
# 3.1. データの前処理

# データの前処理
def preprocess_sentence(sentence, vocab):
    # 単語をIDに変換
    sentence = [vocab[word] if word in vocab else vocab['<unk>'] for word in sentence.split()]
    return sentence

# 前処理を実行
train_data['Q'] = train_data['Q'].apply(lambda x: preprocess_sentence(x, vocab))

# 3.2. データのローダーの作成

# データのローダーの作成
class DataLoader():
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = 0
        self.max_index = np.ceil(len(self.data) / self.batch_size).astype(np.int)
        if self.shuffle:
            self.data = self.data.sample(frac=1)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= self.max_index:
            self.index = 0
            if self.shuffle:
                self.data = self.data.sample(frac=1)
            raise StopIteration()
        batch = self.data.iloc[self.index*self.batch_size:(self.index+1)*self.batch_size]
        self.index += 1
        return batch
    
    def __len__(self):
        return self.max_index
    
    def reset(self):
        self.index = 0

# データのローダーを作成
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

# 4. モデルの構築
# 4.1. パラメータの設定

# パラメータの設定
BATCH_SIZE = 64
EMBED_SIZE = 512
HIDDEN_SIZE = 512
NUM_LAYERS = 2
NUM_HEADS = 8
DROPOUT = 0.1
MAX_LENGTH = 50
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
TEACHER_FORCING_RATIO = 0.5


# 4.2. モデルの構築

# モデルの構築
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([EncoderLayer(embed_size, hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        x = self.embed(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, embed_size, hidden_size, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttentionLayer(embed_size, num_heads, dropout)
        self.feed_forward = FeedForwardLayer(embed_size, hidden_size, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        x2 = self.self_attention(x, x, x, mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.feed_forward(x)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x
    
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        self.linear_q = nn.Linear(embed_size, embed_size)
        self.linear_k = nn.Linear(embed_size, embed_size)
        self.linear_v = nn.Linear(embed_size, embed_size)
        self.linear = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        q = q.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(self.head_size)
        scores.masked_fill_(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        x = torch.matmul(attention, v)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.embed_size)
        x = self.linear(x)
        return x

class FeedForwardLayer(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(embed_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([DecoderLayer(embed_size, hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_outputs, src_mask, trg_mask):
        x = self.embed(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_outputs, src_mask, trg_mask)
        x = self.linear(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, embed_size, hidden_size, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttentionLayer(embed_size, num_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(embed_size, num_heads, dropout)
        self.feed_forward = FeedForwardLayer(embed_size, hidden_size, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_outputs, src_mask, trg_mask):
        x2 = self.self_attention(x, x, x, trg_mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.encoder_attention(x, encoder_outputs, encoder_outputs, src_mask)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        x2 = self.feed_forward(x)
        x = x + self.dropout3(x2)
        x = self.norm3(x)
        return x
    
# Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx):
        super(Seq2Seq, self).__init__() 
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        encoder_outputs = self.encoder(src, src_mask)
        output = self.decoder(trg, encoder_outputs, src_mask, trg_mask)
        return output
    
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 2048
DEC_PF_DIM = 2048
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

## The model has 29,320,449 trainable parameters

optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:,:-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

## Epoch: 01 | Time: 0m 48s
## 	Train Loss: 5.113 | Train PPL: 165.000
## 	 Val. Loss: 5.201 |  Val. PPL: 182.000
## Epoch: 02 | Time: 0m 48s
## 	Train Loss: 4.545 | Train PPL:  93.000
## 	 Val. Loss: 4.983 |  Val. PPL: 145.000
## Epoch: 03 | Time: 0m 48s
## 	Train Loss: 4.201 | Train PPL:  66.000
## 	 Val. Loss: 4.800 |  Val. PPL: 122.000
## Epoch: 04 | Time: 0m 48s
## 	Train Loss: 3.952 | Train PPL:  52.000
## 	 Val. Loss: 4.665 |  Val. PPL: 106.000 
## Epoch: 05 | Time: 0m 48s
## 	Train Loss: 3.757 | Train PPL:  43.000
## 	 Val. Loss: 4.569 |  Val. PPL:  96.000
## Epoch: 06 | Time: 0m 48s
## 	Train Loss: 3.590 | Train PPL:  36.000
## 	 Val. Loss: 4.500 |  Val. PPL:  90.000
## Epoch: 07 | Time: 0m 48s
## 	Train Loss: 3.442 | Train PPL:  31.000
## 	 Val. Loss: 4.456 |  Val. PPL:  86.000
## Epoch: 08 | Time: 0m 48s
## 	Train Loss: 3.309 | Train PPL:  27.000
## 	 Val. Loss: 4.428 |  Val. PPL:  84.000
## Epoch: 09 | Time: 0m 48s
## 	Train Loss: 3.188 | Train PPL:  24.000
## 	 Val. Loss: 4.415 |  Val. PPL:  83.000
## Epoch: 10 | Time: 0m 48s
## 	Train Loss: 3.076 | Train PPL:  21.000
## 	 Val. Loss: 4.411 |  Val. PPL:  83.000

model.load_state_dict(torch.load('tut6-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

## | Test Loss: 4.411 | Test PPL:  83.000 |

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    model.eval()
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    mask = model.create_mask(src_tensor)
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, _ = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:]

def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
    assert n_rows * n_cols == n_heads
    fig = plt.figure(figsize=(15,25))
    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()
        cax = ax.matshow(_attention, cmap='bone')
        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + ['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                           rotation=45)
        ax.set_yticklabels([''] + translation)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    plt.close()

example_idx = 12

src = vars(train_data.examples[example_idx])['src']
trg = vars(train_data.examples[example_idx])['trg']

print(f'src = {src}')

print(f'trg = {trg}')

## src = ['ein', 'mädchen', 'in', 'einer', 'blauen', 'jacke', 'und', 'einem', 'hut', ',', 'das', 'auf', 'einem', 'fahrrad', 'steht', '.']


