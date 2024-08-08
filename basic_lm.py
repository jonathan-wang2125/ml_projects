import math

import os
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt

from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader


os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import torch

device = torch.device("cuda:0")
print("Using device:", device)

path = '/afs/csail.mit.edu/u/j/jwang27/.cache/torch/text/datasets/PennTreebank'
print(os.path.exists(path))

# Initialize the tokenizer
tokenizer = get_tokenizer("basic_english")

# Load the WikiText2 dataset
train_iter, val_iter, test_iter = PennTreebank()

# Define a function to yield tokens from the dataset
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# Build the vocabulary from the training dataset
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Define a function to process the raw text and convert it to tensors
def data_process(raw_text_iter):
    data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# Process the train, validation, and test datasets
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

# Define a function to batchify the data
def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data

# Set the batch sizes and batchify the data
batch_size = 10
eval_batch_size = 20
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

# Define the TransformerModel class
class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
##Generate a mask for the input sequence
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
## Change all the zeros to negative infinity and all the ones to zeros as follows:
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
# Define the forward pass
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
# Define the PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

ntokens = len(vocab)  # size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # dimension of the feedforward network model in ``nn.TransformerEncoder``

control_hyperparams = {
    'batch_size': 50,
    'epochs': 15,
    'lr': 0.001,
    'nlayers': 2,
    'nhead': 1,
    'dropout': 0.1,
    'scheduler': 'StepLR'
}


hyperparams = {
    'batch_size': [50, 100, 150, 200, 250],
    'epochs': [5, 10, 15, 20, 25],
    'lr': [0.001, 0.005, 0.01, 0.05, 0.1],
    'nlayers': [1, 2, 3, 4, 5],
    'nhead': [1, 2, 4, 10, 20],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    'scheduler': ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau']
}

def get_batch(source, i, bptt=35):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data.to(device), target.to(device)

def train(model, train_data, criterion, optimizer, batch_size):
    model.train()  # Set the model to training mode
    total_loss = 0.  # Initialize the total loss to 0
    ntokens = len(vocab)  # Get the number of tokens in the vocabulary

    # Iterate through the mini-batches of data
    for batch, i in enumerate(range(0, train_data.size(0) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)  # Get the input data and targets for the current mini-batch
        optimizer.zero_grad()  # Reset the gradients to zero before the next backward pass
        output = model(data)  # Forward pass: compute the output of the model given the input data
        loss = criterion(output.view(-1, ntokens), targets)  # Calculate the loss between the model output and the targets
        loss.backward()  # Backward pass: compute the gradients of the loss with respect to the model parameters
        optimizer.step()  # Update the model parameters using the computed gradients
        total_loss += loss.item()  # Accumulate the total loss

    return total_loss / (batch + 1)  # Return the average loss per mini-batch


def evaluate(model, data_source, criterion, batch_size, bptt=35):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.  # Initialize the total loss to 0
    ntokens = len(vocab)  # Get the number of tokens in the vocabulary

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)  # Get the input data and targets for the current mini-batch
            data, targets = data.to(device), targets.to(device)
            output = model(data)  # Forward pass: compute the output of the model given the input data
            loss = criterion(output.view(-1, ntokens), targets)  # Calculate the loss between the model output and the targets
            total_loss += loss.item()  # Accumulate the total loss

    return total_loss / (i + 1)  # Return the average loss per mini-batch




def get_scheduler(optimizer, scheduler_name):
    if scheduler_name == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif scheduler_name == 'MultiStepLR':
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)
    elif scheduler_name == 'ExponentialLR':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_name == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif scheduler_name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


fig, axs = plt.subplots(len(hyperparams), 1, figsize=(10, 28))

# Iterate over each hyperparameter
for i, (param, values) in enumerate(hyperparams.items()):
    val_losses = []

    for value in values:
        hyperparam_dict = control_hyperparams.copy()
        hyperparam_dict[param] = value

        batch_size = hyperparam_dict['batch_size']
        train_data_batchified = batchify(train_data, batch_size)
        val_data_batchified = batchify(val_data, batch_size)

        model = TransformerModel(ntokens, emsize, hyperparam_dict['nhead'], nhid, hyperparam_dict['nlayers'], hyperparam_dict['dropout']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparam_dict['lr'])
        scheduler = get_scheduler(optimizer, hyperparam_dict['scheduler'])

        best_val_loss = float('inf')
        for epoch in range(hyperparam_dict['epochs']):
            train_loss = train(model, train_data_batchified, criterion, optimizer, batch_size)
            val_loss = evaluate(model, val_data_batchified, criterion, batch_size)
            print(f"Epoch {epoch+1}/{hyperparam_dict['epochs']}, {param}: {value}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        val_losses.append(best_val_loss)

    axs[i].plot(values, val_losses, label=f'Validation Loss')
    axs[i].set_xlabel(param)
    axs[i].set_ylabel('Loss')
    axs[i].set_title(f'Hyperparameter: {param}')
    axs[i].legend()

plt.tight_layout()
plt.show()
