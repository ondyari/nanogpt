import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl

import os
import wget
import argparse
from argparse import Namespace

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*srun python.*")



# ------------ Hyperparameters ------------

def get_parser():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('--embedding_size', type=int, default=384)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--num_blocks', type=int, default=6)
    parser.add_argument('--dropout_prob', type=float, default=0.2)

    # Training arguments
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--eval_iters', type=int, default=200)

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--data_split', type=float, default=0.9, help='Percentage of data to use for training')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)

    return parser

# ------------ Data ------------

class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.args.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.args.block_size]
        y = self.data[idx+1:idx+self.args.block_size+1]
        return x, y


class ShakespeareDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self, stage=None):
        print("Preparing data")
        # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
        # Download the data if required
        input_file = os.path.join(self.args.data_dir, 'input.txt')
        if not os.path.exists(input_file):
            os.makedirs(self.args.data_dir, exist_ok=True)
            wget.download('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',  out=self.args.data_dir)

        with open(input_file, 'r', encoding='utf-8') as f:
            self.text = f.read()

        # here are all the unique characters that occur in this text
        self.chars = sorted(list(set(self.text)))
        # create a mapping from characters to integers
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        data = np.array(self.encode(self.text), dtype=np.int64)
        n = int(self.args.data_split * len(data))
        self.train_dataset = ShakespeareDataset(self.args, data[:n])
        self.val_dataset = ShakespeareDataset(self.args, data[n:])
    
    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
    
    def get_vocab_size(self):
        return len(self.stoi)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)


# ------------ Architecture ------------

class Head(nn.Module):
    def __init__(self, input_channels, size=16, block_size=8, dropout_prob=0.0) -> None:
        super().__init__()
        self.size = size
        self.key = nn.Linear(input_channels, size, bias=False)
        self.query = nn.Linear(input_channels, size, bias=False)
        self.value = nn.Linear(input_channels, size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # shape: (B, T, size)
        q = self.query(x)  # shape: (B, T, size)
        wei = torch.einsum('bth,bTh->btT', q, k)  # shape: (B, T, T)
        wei *= C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)  # shape: (B, T, size)
        out = torch.einsum('btT,bTh->bth', wei, v)  # shape: (B, T, size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, input_channels, size=16, block_size=8, nheads=1, dropout_prob=0.0) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(input_channels, size, block_size, dropout_prob=dropout_prob) for _ in range(nheads)])
        self.proj = nn.Linear(nheads*size, input_channels)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, size, dropout_prop=0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, 4 * size),
            nn.ReLU(),
            nn.Linear(4 * size, size),
            nn.Dropout(dropout_prop),
        )
    
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, size, nheads=4, block_size=8, dropout_prob=0.0) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(size, size//nheads, block_size, nheads, dropout_prob)
        self.ff = FeedForward(size, dropout_prob)
        self.ln1 = nn.LayerNorm(size)
        self.ln2 = nn.LayerNorm(size)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    
    """
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x
    """


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6) -> None:
        super().__init__()
        self.size = size
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.eps) + self.bias


# super simple bigram model
class TransformerModel(nn.Module):

    def __init__(self, embedding_size, vocab_size, block_size, num_blocks=4, dropout_prob=0.0, num_heads=4):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(block_size, embedding_size)
        assert embedding_size % num_heads == 0
        self.blocks = nn.Sequential(
            *[Block(embedding_size, num_heads, block_size, dropout_prob=dropout_prob) for _ in range(num_blocks)],
            nn.LayerNorm(embedding_size)
        )
        self.lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=tok_emb.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C) 
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# ------------ Model ------------

class NanoGPT(pl.LightningModule):
    def __init__(self, args, vocab_size):
        super().__init__()
        self.args = args
        self.vocab_size = vocab_size

        self.save_hyperparameters()

        self.model = TransformerModel(args.embedding_size, vocab_size=vocab_size, block_size=args.block_size, num_blocks=args.num_blocks, dropout_prob=args.dropout_prob, num_heads=args.num_heads) 
    
    def forward(self, idx, targets=None):
        return self.model(idx, targets=targets)

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.args.block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits, loss = self(idx, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        idx, targets = batch
        logits, loss = self(idx, targets)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        return optimizer

# ------------ Training ------------

def train(args):
    # Fix seed for reproducibility
    torch.manual_seed(1337)
    np.random.seed(1337)

    shakespeare_data = ShakespeareDataModule(args)
    shakespeare_data.prepare_data()

    model = NanoGPT(args, vocab_size=shakespeare_data.get_vocab_size())

    logger = pl.loggers.TensorBoardLogger('lightnight_logs')

    trainer = pl.Trainer(
        enable_checkpointing=False,
        max_steps=args.max_iters,
        limit_val_batches=args.eval_iters,
        val_check_interval=args.eval_interval,
        logger=logger,
    )
    trainer.fit(model, shakespeare_data)
    trainer.save_checkpoint('final.ckpt')

    # Generate some text
    context = torch.zeros((1, 1), dtype=torch.long, device=model.device)
    print(shakespeare_data.decode(model.generate(context, max_new_tokens=1000)[0].tolist()))
    with open('generated.txt', 'w') as f:
        f.write(shakespeare_data.decode(model.generate(context, max_new_tokens=1000)[0].tolist()))


# ------------ Main ------------

if __name__ == '__main__':
    args = get_parser().parse_args()
    print(args)

    train(args)

