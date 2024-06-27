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
import os
import wandb


# TODO : make sure data generated is as needed
#        label each model and result accurately
#        note model arch
#        make everything run in one go; training and results for each seed

# np.random.seed(879965)
# np.random.seed(10020)
# np.random.seed(100201)
# np.random.seed(1005694)
np.random.seed(789221)



class AlternatingSignalDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

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

# def visualize_signals(input_signal, target_signal):
#     plt.figure(figsize=(12, 6))
    
#     # plt.subplot(2, 1, 1)
#     plt.plot(input_signal.numpy(), 'r--', drawstyle='steps-post')
#     plt.title('Input Signal')
    
#     # plt.subplot(2, 1, 2)
#     plt.plot(target_signal.numpy(), 'b--', drawstyle='steps-post')
#     plt.title('Target Signal')
    
#     plt.tight_layout()
#     plt.show()



# Split train data into train and validation sets
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


hld_ot_int = np.random.randint(0, 99, size=10).tolist()
print(hld_ot_int)
# Generate data with holdout intervals
train_inputs, train_targets, test_inputs, test_targets = generate_inter_trial_interval(1000, 200, holdout_intervals=hld_ot_int)

train_inputs_split, train_targets_split, val_inputs_split, val_targets_split = split_train_val(train_inputs, train_targets, val_split=0.1)


# Save tensors to a file
torch.save({'inputs': test_inputs, 'targets': test_targets}, 'test_tensors.pth')


train_set = AlternatingSignalDataset(train_inputs_split, train_targets_split)
val_set = AlternatingSignalDataset(val_inputs_split, val_targets_split)
test_set = AlternatingSignalDataset(test_inputs, test_targets)


# def visualize_signals(input_signal, target_signal):
#     plt.figure(figsize=(12, 6))
    
#     plt.subplot(2, 1, 1)
#     plt.plot(input_signal.numpy(), 'r--', drawstyle='steps-post')
#     plt.title('Input Signal')
    
#     plt.subplot(2, 1, 2)
#     plt.plot(target_signal.numpy(), 'b--', drawstyle='steps-post')
#     plt.title('Target Signal')
    
#     plt.tight_layout()
#     plt.show()

# # Visualize a sample from the training data
# visualize_signals(train_inputs_split[0], train_targets_split[0])

# Visualize a sample from the holdout data
# visualize_signals(test_inputs[0], test_targets[0])

print('\n')
print(f'train inputs split shape: {train_inputs_split.shape}')
print(f'train targets split shape: {train_targets_split.shape}')
print(f'val inputs split shape: {val_inputs_split.shape}')
print(f'val targets split shape: {val_targets_split.shape}')
print(f'test inputs  shape: {test_inputs.shape}')
print(f'test targets  shape: {test_targets.shape}')
print('\n')

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
    # print(name)
    plt.savefig(save_path)
    # plt.show()


# split dataset into train and validation sets
def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val


class LSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=8,
            num_layers=4,
            batch_first=True,
            )
        
        self.linear = nn.Linear(8, 1)


    def forward(self, x):

        x, _ = self.lstm(x)
        x = self.linear(x)

        return x
    


# load data
batch_size = 32
num_workers = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# # print(test_x)
# for i in range(10):

#     print('='*10, f'\n {test_x[i].T} \n')

#     # visualize_signals(test_x[i], test_y[i], i)




# train_set = AlternatingSignalDataset(train_x, train_y)
# val_set = AlternatingSignalDataset(val_x, val_y)

trainset, _ = split_train_val(test_set, val_split=0.2)
_, valset = split_train_val(test_set, val_split=0.1)
test_set, _ = split_train_val(test_set, val_split=0)



# Dataloaders
trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valloader = DataLoader(
    valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = DataLoader(
    test_set, batch_size=1, shuffle=False, num_workers=num_workers)


model = LSTM()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

epochs = 1000

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True




# wandb.init(
#     project="LSTM signals",
#     config={pbar = tqdm(range(start_epoch, epochs))
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
#         "learning_rate": 0.001,
#         "architecture": "LSTM",
#         "train_set": test_set,
#         "epochs": epochs,
#     }
# )


# train model

def train():
    model.train()
    train_loss = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))

    for batch_i, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        total += targets.size(0)

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f' %
            (batch_i, len(trainloader), train_loss/(batch_i+1))
        )

        wandb.log({"train_loss": train_loss / (batch_i + 1), "epoch": epoch})


def eval(dataloader):

    model.eval()
    eval_loss = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(valloader))

        for batch_ind, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            total += inputs.size(0)
            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f' %
                (batch_ind, len(dataloader), eval_loss/(batch_ind+1))
            )

            wandb.log({"val_loss": eval_loss / (batch_ind + 1), "epoch": epoch})

seeds = [879965, 10020, 100201, 1005694, 789221]


# for model_it in range(5):


# wandb.init(
#     project="LSTM",
#     config={
#         "learning_rate": 0.001,
#         "architecture": "lstm",
#         "train_set": train_set,
#         "epochs": epochs,
#     }
# )

# pbar = tqdm(range(0, epochs))
# for epoch in pbar:

#     pbar.set_description('Epoch: %d' % (epoch))
#     train()
#     eval(valloader)

#     if not os.path.isdir('checkpoint'):
#         os.mkdir('checkpoint')
#     checkpoint_path = f'./checkpoint/lstm_signal_testing_ckpt_{4}.pth'

#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#     }, checkpoint_path)

# wandb.finish()

# print(f'Number of batches: {len(testloader)}')
# for batch_idx, (inputs, targets) in enumerate(testloader):
#     print(f'Batch {batch_idx} - Input shape: {inputs.shape}, Target shape: {targets.shape}')


# with torch.no_grad():
#         pbar = tqdm(enumerate(testloader))
#         ctr = 0
#         for batch_idx, (inputs, targets) in pbar:
#             print(batch_idx)
#             inputs, targets = inputs.to(device), targets.to(device)

#             # visualize_signals(targets[batch_idx].cpu().detach(), output, ctr)
#             visualize_signals(inputs[batch_idx][:10].cpu().detach(), targets[batch_idx][:10].cpu().detach(), ctr)
#             ctr+=1


# checkpoint = torch.load('./checkpoint/lstm_signal_testing_ckpt.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']

# with torch.no_grad():
#         pbar = tqdm(enumerate(testloader))
#         ctr = 0
#         for batch_idx, (inputs, targets) in pbar:
#             inputs, targets = inputs.to(device), targets.to(device)
#             print(inputs.shape)
#             outputs = torch.sigmoid(model(inputs).squeeze().cpu().detach())
#             print(outputs.shape)
#             for output in outputs:
#                 print(inputs[batch_idx])
#                 # visualize_signals(targets[batch_idx].cpu().detach(), output, ctr)
#                 visualize_signals(inputs[batch_idx].cpu().detach(), targets[batch_idx].cpu().detach(), ctr)
#                 ctr+=1





# TODO fix make into one loop code weirfd an messy


# test_ckpt = torch.load('test_tensors.pth')
# test_inputs = test_ckpt['inputs']
# test_targets = test_ckpt['targets']

# for ind, seed in enumerate(seeds):

#     print('#'*10)
#     print(f'\nworking on model {ind}, with seed {seed}...')
#     np.random.seed(seed)

#     print(f'\nloading checkpoint lstm_signal_testing_ckpt_{ind}.pth\n')
#     checkpoint = torch.load(f'./checkpoint/lstm_signal_testing_ckpt_{ind}.pth')
#     print(f'loading model state dict...\n')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     print(f'loading optimizer state dict...\n')
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#     epoch = checkpoint['epoch']


#     print(test_inputs.shape)
#     print(test_targets.shape)
    
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
#     for out in range(5):nph}/{name[1]}.png'

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
