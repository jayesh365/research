'''
Train S4D model 

'''

import torch
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from models.s4.s4d import S4D
from tqdm.auto import tqdm


import numpy as np
import matplotlib.pyplot as plt
import time

import wandb

import os

np.random.seed(747208)


class AlternatingSignalDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# generate data 
def generate_alternating_signal(every_n, n, num_seq=100):
    '''
    generate data to train S4D model
    data is of the form of 1|0 every n time steps for a total length of n
    if every_n = 5 then the TS can be 000001111100000...
    however the first signal does not have to be of length every_n
    i.e if every_n = 5, we can get a TS of 00111110000011111...
    '''

    inputs = []
    targets = []

    for seq_num in range(num_seq):
        # pick starting number for inital singal
        start = np.random.randint(low=1, high=every_n)

        remaining_length = n+1 - start

        # define the number of sections for each signal
        num_sections = int(np.ceil(remaining_length / every_n))

        # create num_sections of signals
        section = torch.concat((torch.ones(5), torch.zeros(5)))

        # create time series
        signal = section.repeat(num_sections)
        ts = signal[:remaining_length]
        ts = torch.concat((torch.zeros(start), ts))


        # Create input and target sequences
        input_seq = ts[1:].unsqueeze(-1)  # Exclude the last element for input
        target_seq = ts[:-1].unsqueeze(-1)  # Exclude the first element for target

        inputs.append(input_seq)
        targets.append(target_seq)


    return torch.stack(inputs), torch.stack(targets)


def visualize_signals(input_signal, output_signal, ind):
    plt.figure(figsize=(14, 6))

    # Plot input signal
    plt.plot(input_signal.numpy().flatten(), linestyle='dashed', drawstyle='steps-post', label='Target Signal')
    plt.title('Target Signal')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.plot(output_signal.numpy().flatten(), linestyle='dashed', drawstyle='steps-post', label='Output Signal', color='green')
    plt.title('Output Signal')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./outputs/s4_{ind}.png')
    plt.show()


# split dataset into train and validation sets
def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

d_input = 1
d_output = 100
d_model = 128
n_layers = 4
dropout = 0.2
prenorm = False

epochs = 30


train_input, train_target = generate_alternating_signal(5, 100, 10000)
val_input, val_target = generate_alternating_signal(5, 100, 10000)
test_input, test_target = generate_alternating_signal(5, 100, 10)

train_set = AlternatingSignalDataset(train_input, train_target)
val_set = AlternatingSignalDataset(val_input, val_target)
test_set = AlternatingSignalDataset(test_input, test_target)

trainset, _ = split_train_val(train_set, val_split=0.1)
_, valset = split_train_val(val_set, val_split=0.1)
batch_size = 64
num_workers = 4


# Dataloaders
trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
valloader = DataLoader(
    valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
testloader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)




wandb.init(
    project="s4 custom signals",
    config={
        "learning_rate": 0.01,
        "architecture": "S4D",
        "train_set": train_set,
        "epochs": epochs,
    }
)

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


# define S4D model
class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=1,
        d_model=128,
        n_layers=1,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        # before Encoder torch.Size([64, 100, 1])
        # print('\n shape X: ', x.shape, '\n')

        # after Encoder torch.Size([64, 100, 128])
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        # print('\n shape encoder X: ', x.shape, '\n')

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length

        x = x.mean(dim=1)
        

        # print('\n shape output X: ', x.shape, '\n')
        
        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        # print('\n shape output Decoder X: ', x.shape, '\n')

        # print('\n debug output ', x, '\n')
        return x


# Model
# print('==> Building model..')
model = S4Model(
    d_input=d_input,
    d_output=d_output,
    d_model=d_model,
    n_layers=n_layers,
    dropout=dropout,
    prenorm=prenorm,
)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True


def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # print('all parameters: ', all_parameters)

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss(reduction='sum')
optimizer, scheduler = setup_optimizer(
    model, lr=0.01, weight_decay=0.01, epochs=epochs
)

# ###############################################################################
# # Everything after this point is standard PyTorch training!
# ###############################################################################


# check if A matrix is changing for each epoch
A_matrices = {}
def does_A_change(model, epoch):

    for layer, name in enumerate(model.s4_layers):
        print('\nepoch_' + str(epoch) + '_layer_' + str(layer))
        log_A_real = model.s4_layers[layer].kernel.log_A_real
        A_imag = model.s4_layers[layer].kernel.A_imag
        A_matrices['epoch_' + str(epoch) + '_layer_' + str(layer)] = -torch.exp(log_A_real) + 1j * A_imag
        
# Training
def train():
    model.train()
    train_loss = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))

    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).unsqueeze(-1)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1))
        )

        wandb.log({"train_loss": train_loss / (batch_idx + 1), "epoch": epoch})




def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).unsqueeze(-1)
            # need to apply sigmoid
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += inputs.size(0)
            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1))
            )

            wandb.log({"val_loss": eval_loss / (batch_idx + 1), "epoch": epoch})


pbar = tqdm(range(start_epoch, epochs))
for epoch in pbar:

    pbar.set_description('Epoch: %d' % (epoch))

    train()
    eval(epoch, testloader)
    scheduler.step() 
       
    print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    checkpoint_path = './checkpoint/signal_testing_ckpt.pth'

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

wandb.finish()


checkpoint = torch.load('./checkpoint/signal_testing_ckpt.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

with torch.no_grad():
        pbar = tqdm(enumerate(testloader))
        ctr = 0
        for batch_idx, (inputs, targets) in pbar:
            # print(f'\nBatch index: {batch_idx}')
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs = inputs.unsqueeze(0)
            outputs = torch.sigmoid(model(inputs).unsqueeze(-1).cpu().detach())
            # print(outputs.shape)
            for output in outputs:
                # visualize_signals(inputs[batch_idx].cpu().detach(), targets[batch_idx].cpu().detach())
                visualize_signals(targets[batch_idx].cpu().detach(), output, ctr)
                ctr+=1
