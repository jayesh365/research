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


# np.random.seed(879965)
# np.random.seed(10020)
# np.random.seed(100201)
# np.random.seed(1005694)
# np.random.seed(789221)

class AlternatingSignalDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# generate data 
def generate_alternating_signal(every_n, n, num_seq=100, custom_start_sig=None, custom_start=None):
    '''
    generate data to train S4D model
    data is of the form of 1|0 every n time steps for a total length of n
    if every_n = 5 then the TS can be 000001111100000...
    however the first signal does not have to be of length every_n
    i.e if every_n = 5, we can get a TS of 00111110000011111...
    '''

    inputs = []
    targets = []

    for _ in range(num_seq):
        # pick starting number for inital singal
        if custom_start == None: 
            # print('start should not print')
            start = np.random.randint(low=1, high=every_n)
        else: start = custom_start
        
        if custom_start_sig == None: 
            starting_sig=np.random.randint(2, size=1)[0]
        else: starting_sig = custom_start_sig

        # print('='*3, f'\nstart {starting_sig} number of {start}.\n')

        remaining_length = n+1 - start


        # define the number of sections for each signal
        num_sections = int(np.ceil(remaining_length / every_n))
    
        # create num_sections of signals
        if starting_sig == 1: section = torch.concat((torch.zeros(5), torch.ones(5)))
        else: section = torch.concat((torch.ones(5), torch.zeros(5)))

        # create time series
        signal = section.repeat(num_sections)
        ts = signal[:remaining_length]
        if starting_sig == 0: ts = torch.concat((torch.zeros(start), ts))
        else: ts = torch.concat((torch.ones(start), ts))


        # Create input and target sequences
        input_seq = ts[1:].unsqueeze(-1)  # Exclude the last element for input
        target_seq = ts[:-1].unsqueeze(-1)  # Exclude the first element for target

        inputs.append(input_seq)
        targets.append(target_seq)
        # print('DEBIG: ', custom_start, custom_start_sig, remaining_length, num_sections)

    # for i in range(len(inputs)): print('signal: ', inputs[i].T, '='*3)


    return torch.stack(inputs), torch.stack(targets)




def visualize_signals_og(input_signal, output_signal, name, col, save_path):
    plt.figure(figsize=(14, 6))

    # Plot input signal
    plt.plot(input_signal.numpy().flatten(), linestyle='dashed', drawstyle='steps-post', label='Target Signal', color=col)
    plt.title('Target Signal')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.plot(output_signal.numpy().flatten(), linestyle='dashed', drawstyle='steps-post', label='Output Signal', color='red')
    plt.title('Output Signal')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.title(name[0])

    plt.tight_layout()
    print(name)
    # plt.savefig(f'./outputs/s4d2/s4d1/{name[1]}.png')
    plt.savefig(save_path)

    # plt.show()


# # split dataset into train and validation sets
# def split_train_val(train, val_split):
#     train_len = int(len(train) * (1.0-val_split))
#     train, val = torch.utils.data.random_split(
#         train,
#         (train_len, len(train) - train_len),
#         generator=torch.Generator().manual_seed(42),
#     )
#     return train, val

def split_train_val(train_inputs, train_targets, val_split=0.1):
    dataset_size = len(train_inputs)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_inputs_split = train_inputs[train_indices]
    train_targets_split = train_targets[train_indices]
    
    val_inputs_split = train_inputs[val_indices]
    val_targets_split = train_targets[val_indices]
    
    return train_inputs_split, train_targets_split, val_inputs_split, val_targets_split


def generate_inter_trial_interval(num_seq=1000, n=200, holdout_intervals=None):
    '''
    Generate data to train S4D model.
    Data is of the form: initially off for (20-40) time steps, 
    then on for 10 time steps, and repeats.
    The length of the signals should be 200 time steps.
    '''
    holdout_inputs = []
    holdout_targets = []
    inputs = []
    targets = []


    for seq_num in range(num_seq):
        signal = []
        length = 0
        
        while length < n:
            off_length = np.random.randint(20, 41)
            on_length = 10
            signal.extend([0] * off_length)
            signal.extend([1] * on_length)
            length += off_length + on_length

        signal = signal[:n]
        ts = torch.tensor(signal, dtype=torch.float32).unsqueeze(-1)
        
        # Create input and target sequences
        input_seq = ts
        target_seq = torch.cat((ts[1:], torch.tensor([0], dtype=torch.float32).unsqueeze(-1)), dim=0)
        
        # Check if the current sequence is part of the holdout intervals
        if holdout_intervals and seq_num in holdout_intervals:
            holdout_inputs.append(input_seq)
            holdout_targets.append(target_seq)
        else:
            inputs.append(input_seq)
            targets.append(target_seq)

    return torch.stack(inputs), torch.stack(targets), torch.stack(holdout_inputs), torch.stack(holdout_targets)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

d_input = 1
d_output = 200
d_model = 8
n_layers = 4
dropout = 0
prenorm = False

epochs = 1000

batch_size = 32
num_workers = 4


hld_ot_int = np.random.randint(0, 99, size=10).tolist()
# print(hld_ot_int)
# Generate data with holdout intervals
train_inputs, train_targets, test_inputs, test_targets = generate_inter_trial_interval(1000, 200, holdout_intervals=hld_ot_int)

train_inputs_split, train_targets_split, val_inputs_split, val_targets_split = split_train_val(train_inputs, train_targets, val_split=0.1)


train_set = AlternatingSignalDataset(train_inputs_split, train_targets_split)
val_set = AlternatingSignalDataset(val_inputs_split, val_targets_split)
test_set = AlternatingSignalDataset(test_inputs, test_targets)


def visualize_signals(input_signal, target_signal):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(input_signal.numpy(), 'r--', drawstyle='steps-post')
    plt.title('Input Signal')
    
    plt.subplot(2, 1, 2)
    plt.plot(target_signal.numpy(), 'b--', drawstyle='steps-post')
    plt.title('Target Signal')
    
    plt.tight_layout()
    plt.show()

# # Visualize a sample from the training data
# visualize_signals(train_inputs_split[0], train_targets_split[0])

# Visualize a sample from the holdout data
# visualize_signals(test_inputs[1], test_targets[1])


print(f'train inputs split shape: {train_inputs_split.shape}')
print(f'train targets split shape: {train_targets_split.shape}')
print(f'val inputs split shape: {val_inputs_split.shape}')
print(f'val targets split shape: {val_targets_split.shape}')
print(f'test inputs  shape: {test_inputs.shape}')
print(f'test targets  shape: {test_targets.shape}')




# # print('\n', '='*20, '\nTRAIN\n', '='*20, '\n')
# train_x, train_y = generate_alternating_signal(5, 100, 10000, custom_start=None, custom_start_sig=None)
# # print('\n', '='*20, '\nVAL\n', '='*20, '\n')

# val_x, val_y = generate_alternating_signal(5, 100, 10000, custom_start=None, custom_start_sig=None)

# # print('\n', '='*20, '\nTest\n', '='*20, '\n')

# test_x, test_y= torch.empty(0), torch.empty(0)

# for i in range(5):
#     if test_x.numel() == 0: test_x, test_y = generate_alternating_signal(5, 100, 1, custom_start_sig=0, custom_start=+1)
#     else: 
#         new_x, new_y = generate_alternating_signal(5, 100, 1, custom_start_sig= 0, custom_start= i+1)

#         test_x, test_y = torch.concat((test_x, new_x)), torch.concat((test_y, new_y))

# for i in range(5):
#     new_x, new_y = generate_alternating_signal(5, 100, 1, custom_start_sig= 1, custom_start= i+1)
#     test_x, test_y = torch.concat((test_x, new_x)), torch.concat((test_y, new_y))

# test_set = AlternatingSignalDataset(test_x, test_y)

# print(test_set.inputs.__len__())

# print(test_x)
# for i in range(10):

#     print('='*10, f'\n {test_x[i].T} \n')

#     visualize_signals(test_x[i], test_y[i], i)


# train_set = AlternatingSignalDataset(train_inputs_split, train_targets_split)
# val_set = AlternatingSignalDataset(val_inputs_split, val_targets_split)
# test_set = AlternatingSignalDataset(test_inputs, test_targets)


# trainset, _ = split_train_val(test_set, val_split=0.2)
# _, valset = split_train_val(test_set, val_split=0.1)
# test_set, _ = split_train_val(test_set, val_split=0)



# Dataloaders
trainloader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valloader = DataLoader(
    val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = DataLoader(
    test_set, batch_size=1, shuffle=False, num_workers=num_workers)

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
        d_model=8,
        n_layers=4,
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

# seeds = [10020]
seeds = [879965, 10020, 100201, 1005694, 789221]


# for model_ind in range(5):

# wandb.init(
#     project="s4 custom signals",
#     config={
#         "learning_rate": 0.001,
#         "architecture": "S4D",
#         "train_set": train_set,
#         "epochs": epochs,
#     }
# )

# pbar = tqdm(range(start_epoch, epochs))
# for epoch in pbar:
#     pbar.set_description('Epoch: %d' % (epoch))

#     train()
#     eval(epoch, testloader)
#     scheduler.step() 
    
#     print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")

#     if not os.path.isdir('checkpoint'):
#         os.mkdir('checkpoint')
#     checkpoint_path = f'./checkpoint/s4d_signal_ckpt_{4}.pth'

#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#     }, checkpoint_path)

# wandb.finish()


# checkpoint = torch.load('./checkpoint/signal_testing_ckpt.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']

# with torch.no_grad():
#         pbar = tqdm(enumerate(testloader))
#         ctr = 0
#         for batch_idx, (inputs, targets) in pbar:
#             # print(f'\nBatch index: {batch_idx}')
#             inputs, targets = inputs.to(device), targets.to(device)
#             # inputs = inputs.unsqueeze(0)
#             outputs = torch.sigmoid(model(inputs).unsqueeze(-1).cpu().detach())
#             # print(outputs.shape)
#             for output in outputs:
#                 # visualize_signals(inputs[batch_idx].cpu().detach(), targets[batch_idx].cpu().detach())
#                 visualize_signals(targets[batch_idx].cpu().detach(), output, ctr)
#                 ctr+=1


# checkpoint = torch.load('./checkpoint/s4d_signal_testing_ckpt_4.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']

# inputs_tensor = test_x.to(device)
# targets_tensor = test_y.to(device)

# # Perform inference
# model.eval()
# with torch.no_grad():
#     print(inputs_tensor.shape)
#     outputs = torch.sigmoid(model(inputs_tensor).squeeze().cpu().detach())

# # # Visualize the results
# for i in range(10):
#     visualize_signals(targets_tensor[i].cpu().detach(), outputs[i].cpu().detach(), i)




# TODO fix make into one loop code weirfd an messy

# i = 0

# checkpoint = torch.load(f'./checkpoint/s4d_signal_ckpt_{i}.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']


# inputs_tensor = test_inputs.to(device)
# targets_tensor = test_targets.to(device)

# on = []
# off = []

# # for i in range(10):
#     # print(inputs_tensor[i].T)

# # Perform inference
# model.eval()
# with torch.no_grad():
#     outputs = torch.sigmoid(model(inputs_tensor).squeeze().cpu().detach())

# # Visualize the results
# for out in range(5):
#     if torch.round(inputs_tensor[out][0]).item() == 0:
#         name = (f'S4D {i} (Input Starts OFF)', f's4d_ckp_{i}_off_{out}')
#         col = 'blue'
#     else: 
#         col = 'green'
#         name = (f'S4D {i} (Input Starts ON)', f's4d_ckp_{i}_on_{out}')


#     # print(col, torch.round(inputs_tensor[out][0]).item())
    

#     if col == 'green': on.append((targets_tensor[out].cpu().detach(), outputs[out].cpu().detach(), name, col))
#     if col == 'blue': off.append((targets_tensor[out].cpu().detach(), outputs[out].cpu().detach(), name, col))

# # for i in range(len(on)):
# #     visualize_signals(on[i][0], on[i][1], on[i][2], on[i][3])

# for i in range(len(off)):
#     visualize_signals(off[i][0], off[i][1], off[i][2], off[i][3])



# for ind, seed in enumerate(seeds):

#     print('#'*10)
#     print(f'\nworking on model {ind}, with seed {seed}...')
#     # np.random.seed(seed)

#     print(f'\nloading checkpoint s4d_signal_ckpt_{ind}.pth\n')
#     checkpoint = torch.load(f'./checkpoint/s4d_signal_ckpt_{ind}.pth')
#     print(f'loading model state dict...\n')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     print(f'loading optimizer state dict...\n')
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#     epoch = checkpoint['epoch']

#     test_ckpt = torch.load('test_tensors.pth')
#     test_inputs = test_ckpt['inputs']
#     test_targets = test_ckpt['targets']
    
#     inputs_tensor = test_inputs.to(device)
#     targets_tensor = test_targets.to(device)

#     on = []
#     off = []

#     print(f'performing inference...\n')
#     # Perform inference
#     model.eval()
#     with torch.no_grad():
#         outputs = torch.sigmoid(model(inputs_tensor).squeeze().cpu().detach())

#     print('visualizing results...\n')
#     # Visualize the results
#     for out in range(5):
#         save_dir_path = f'./outputs/s4d2/s4d{ind}'

#         # make dir if dne
#         if not os.path.isdir(save_dir_path):
#             os.mkdir(save_dir_path)
#         if torch.round(inputs_tensor[out][0]).item() == 0:
#             name = (f'S4D {ind} (Input Starts OFF)', f's4d_ckp_{ind}_off_{out}')
#             col = 'blue'
#         else: 
#             col = 'green'
#             name = (f'S4D {ind} (Input Starts ON)', f's4d_ckp_{ind}_on_{out}')
#         save_path = f'{save_dir_path}/{name[1]}.png'

#         if col == 'green': 
#             # print(col, torch.round(inputs_tensor[out].T))
#             on.append((targets_tensor[out].cpu().detach(), outputs[out].cpu().detach(), name, col, save_path))
#             # print('\n', '='*5)

#         if col == 'blue': 
#             # print(col, torch.round(inputs_tensor[out].T))
#             off.append((targets_tensor[out].cpu().detach(), outputs[out].cpu().detach(), name, col, save_path))

#     # for i in range(len(on)):
#     #     visualize_signals(on[i][0], on[i][1], on[i][2], on[i][3])

#     for i in range(len(off)):
#         visualize_signals_og(off[i][0], off[i][1], off[i][2], off[i][3], off[i][4])
    
#     print(f'finished working on model {ind}, with seed {seed}...\n')