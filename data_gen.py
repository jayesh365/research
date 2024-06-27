import torch
import numpy as np
import matplotlib.pyplot as plt

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
        input_seq = ts[:-1]
        target_seq = ts[1:]
        
        # Check if the current sequence is part of the holdout intervals
        if holdout_intervals and seq_num in holdout_intervals:
            holdout_inputs.append(input_seq)
            holdout_targets.append(target_seq)
        else:
            inputs.append(input_seq)
            targets.append(target_seq)

    return torch.stack(inputs), torch.stack(targets), torch.stack(holdout_inputs), torch.stack(holdout_targets)

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


hld_ot_int = np.random.randint(0, 99, size=10).tolist()
print(hld_ot_int)
# Generate data with holdout intervals 24, 32, and 38
train_inputs, train_targets, test_inputs, test_targets = generate_inter_trial_interval(1000, 200, holdout_intervals=hld_ot_int)

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

train_inputs_split, train_targets_split, val_inputs_split, val_targets_split = split_train_val(train_inputs, train_targets, val_split=0.1)

# Visualize a sample from the training data
visualize_signals(train_inputs_split[0], train_targets_split[0])

# Visualize a sample from the holdout data
visualize_signals(test_inputs[0], test_targets[0])


print(f'train inputs split shape: {train_inputs_split.shape}')
print(f'train targets split shape: {train_targets_split.shape}')
print(f'val inputs split shape: {val_inputs_split.shape}')
print(f'val targets split shape: {val_targets_split.shape}')
print(f'test inputs  shape: {test_inputs.shape}')
print(f'test targets  shape: {test_targets.shape}')