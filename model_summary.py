from torchinfo import summary
import torch
import torch.nn as nn
from lstm_example import LSTM
from custom_example import S4Model
import matplotlib.pyplot as plt
import numpy as np

ind = 0
model = LSTM()
device = 'cuda'

d_input = 1
d_output = 200
d_model = 8
n_layers = 4
dropout = 0
prenorm = False


s4d = S4Model(
    d_input=d_input,
    d_output=d_output,
    d_model=d_model,
    n_layers=n_layers,
    dropout=dropout,
    prenorm=prenorm,
)
print(f'\nloading checkpoint lstm_signal_testing_ckpt_{ind}.pth\n')
checkpoint = torch.load(f'./checkpoint/lstm_signal_testing_ckpt_{ind}.pth')
print(f'loading model state dict...\n')
model.load_state_dict(checkpoint['model_state_dict'])
print(f'loading optimizer state dict...\n')
# optimizer.load_s
# tate_dict(checkpoint['optimizer_state_dict'])


print(f'\nloading checkpoint s4d_signal_ckpt_{ind}.pth\n')
checkpoint = torch.load(f'./checkpoint/s4d_signal_ckpt_{ind}.pth')
print(f'loading model state dict...\n')
s4d.load_state_dict(checkpoint['model_state_dict'])
# print(f'loading optimizer state dict...\n')
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


s4d.cuda()
model.cuda()



# from torchsummary import summary

# Assume your models are instantiated as `lstm_model` and `s4_model`

# Move models to the appropriate device
# model.to(device)
# s4_model.to(device)

# Print summary for LSTM model
# print("LSTM Model Summary:")
# summary(model, input_size=(100, 1))

# Print summary for S4 model
# print("S4 Model Summary:")
# summary(s4d, input_size=(10, 200, 1))


# print('\n\n\n\n')
# print("LSTM Model Summary:")
# summary(model, input_size=(10, 200, 1)) 




def visualize_signals(input_signal, target_signal):
    # plt.figure(figsize=(12, 6))
    
    # plt.subplot(2, 1, 1)
    plt.plot(input_signal.numpy(), 'r--', drawstyle='steps-post')
    # plt.title('Input Signal')
    
    # plt.subplot(2, 1, 2)
    # plt.plot(target_signal.numpy(), 'b--', drawstyle='steps-post')
    # plt.title('Target Signal')

    plt.ylabel('Signal Value', labelpad=20)
    plt.xlabel('Time Step', labelpad=20)



    ax = plt.gca()

    # Remove the top spine (the top part of the box around the plot)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Set the tick parameters to increase the distance for the tick labels
    ax.tick_params(axis='x', pad=10)  # Increase the distance for the x-axis tick labels
    ax.tick_params(axis='y', pad=10)  
    
    # plt.tight_layout()
    plt.show()



test_ckpt = torch.load('test_tensors.pth')
test_inputs = test_ckpt['inputs']
test_targets = test_ckpt['targets']

inputs_tensor = test_inputs.to(device)
targets_tensor = test_targets.to(device)


model.eval()
with torch.no_grad():
    lstm_out = torch.sigmoid(model(inputs_tensor).squeeze().cpu().detach())


s4d.eval()
with torch.no_grad():
    s4d_out = torch.sigmoid(s4d(inputs_tensor).squeeze().cpu().detach())


# visualize_signals(test_inputs[1], test_targets[1])


def generate_custom_target_signals():
    inputs = []
    targets = []

    for onset_len in range(190):
        signal = []
        onset = [0] * onset_len
        on = [1] * 10
        off = [0] * (200 - (onset_len + 10))

        signal.extend(onset)
        signal.extend(on)
        signal.extend(off)

        ts = torch.tensor(signal, dtype=torch.float32)

        # Pad the sequence to 201 time steps
        ts = torch.cat((ts, torch.zeros(1)), dim=0)

        input_seq = ts[:-1].unsqueeze(-1)  # Exclude the last element for input
        target_seq = ts[1:]  # Exclude the first element for target

        inputs.append(input_seq)
        targets.append(target_seq)

    return torch.stack(inputs), torch.stack(targets)




import matplotlib.pyplot as plt

def visualize_signals_comparison(lstm_outputs, s4d_outputs):
    # lstm_outputs and s4d_outputs should be lists of tuples
    # Each tuple should contain (targets, outputs, name, color)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot S4D signals
    for i, (targets, outputs, name, col) in enumerate(s4d_outputs):
        ax = axes[i, 0]
        ax.plot(targets, label='Target', color='blue', linestyle='dashed')
        ax.plot(outputs, label='Output', color='green')
        ax.set_title(f'S4D Model: {name[0]}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Signal Value')
    
    # Plot LSTM signals
    for i, (targets, outputs, name, col) in enumerate(lstm_outputs):
        ax = axes[i, 1]
        ax.plot(targets, label='Target', color='blue', linestyle='dashed')
        ax.plot(outputs, label='Output', color='green')
        ax.set_title(f'LSTM Model: {name[0]}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Signal Value')
    
    # Set overall plot titles and labels
    fig.suptitle('Model Comparison: LSTM vs S4D', fontsize=16)
    axes[0, 0].set_title('S4D Model Signal 1')
    axes[1, 0].set_title('S4D Model Signal 2')
    axes[0, 1].set_title('LSTM Model Signal 1')
    axes[1, 1].set_title('LSTM Model Signal 2')
    
    # Add a single legend for the entire figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Example usage
lstm_outputs = [
    (targets_tensor[1].cpu().detach(), lstm_out[1].cpu().detach(), ('LSTM Signal 1', 'lstm_1'), 'green'),
    (targets_tensor[2].cpu().detach(), lstm_out[2].cpu().detach(), ('LSTM Signal 2', 'lstm_2'), 'green')
]

s4d_outputs = [
    (targets_tensor[1].cpu().detach(), s4d_out[1].cpu().detach(), ('S4D Signal 1', 's4d_1'), 'green'),
    (targets_tensor[2].cpu().detach(), s4d_out[2].cpu().detach(), ('S4D Signal 2', 's4d_2'), 'green')
]

# visualize_signals_comparison(lstm_outputs, s4d_outputs)





# Generate the target signals
input_signals, target_signals = generate_custom_target_signals()

inputs_tensor = input_signals.to(device)
targets_tensor = target_signals.to(device)


print(input_signals.shape, target_signals.shape)


model.eval()
with torch.no_grad():
    lstm_out = torch.sigmoid(model(inputs_tensor).squeeze().cpu().detach())


s4d.eval()
with torch.no_grad():
    s4d_out = torch.sigmoid(s4d(inputs_tensor).squeeze().cpu().detach())



criterion = nn.BCELoss(reduction='none')
errors = criterion(s4d_out.cpu().detach(), target_signals.cpu().detach().squeeze()).numpy()

# Plot errors against onset times for each test case
plt.figure(figsize=(12, 6))
for i in range(10):
    # Extend errors to have 201 elements using the last value
    errors_extended = np.append(errors[i], errors[i][-1])

    # plt.plot(range(201), errors_extended, label=f'loss   {i+1}')
    plt.plot(s4d_out[i], label=f'sd4 {i+1}')
    plt.plot(target_signals[i], label=f'out {i+1}')

plt.xlabel('Onset Time')
plt.ylabel('Error (BCE Loss)')
plt.title('Error vs. Onset Time for Each Test Case')
plt.legend()
plt.show()




# plt.figure(figsize=(15, 10))
# for i in range(len(target_signals)):
#     plt.subplot(10, 21, i+1)
#     # plt.plot(target_signals[i].squeeze().numpy())
#     plt.plot(s4d_out[i].squeeze().numpy())
#     plt.ylim(-0.1, 1.1)
#     plt.axis('off')

# plt.suptitle('Custom Target Signals', fontsize=16)
# plt.show()
