# Plotting functions
import numpy as np
import matplotlib.pyplot as plt

trim_length = 120 #todo: parametrize

#todo: too many similar functions, rewrite this plotting functions better so that only a few are needed
def plot_sample_from_batch(sample_in, sample_out):
    """Show plot of a sample"""
    plt.plot(sample_in, 'r')
    plt.plot(sample_out, 'g')
    #sample_stack = np.vstack((sample_in, sample_out))
    #plt.plot(np.min(sample_stack,0), 'y')
    plt.ylim([0,1])
    plt.xlim([0,trim_length-1])
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def plot_batch(sample_batched):
    """Show plot for a batch of samples."""
    batch_size = len(sample_batched['laser'])
    
    for i in range(batch_size):
        print(i+1, "/", batch_size)
        plt.figure()
        l_in = sample_batched['laser'][i, :]
        t_out = sample_batched['truth'][i, :]
        plot_sample_from_batch(l_in.numpy(), t_out.numpy())
        plt.show()
        
def plot_sample_from_out_batch(sample_in, sample_out, sample_truth):
    """Show plot of a sample"""
    plt.plot(sample_in, 'r')
    plt.plot(sample_out, 'b')
    plt.plot(sample_truth, 'g')
    plt.ylim([0,1])
    plt.xlim([0,trim_length-1])
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def plot_out_batch(in_batched, out_batched, truth_batched):
    """Show plot for a batch of network output."""
    batch_size = len(in_batched)

    for i in range(batch_size):
        print(i+1, "/", batch_size)
        plt.figure()
        l_in = in_batched[i, :]
        nn_out = out_batched[i, :]
        t_out = truth_batched[i, :]
        plot_sample_from_out_batch(l_in.numpy(), nn_out.numpy(), t_out.numpy())
        plt.show()