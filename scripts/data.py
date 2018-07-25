# Dataset and Transformation class definitions
import numpy as np
import torch as tc
import pandas as pd
from torch.utils.data import Dataset

class AugmentMinGroundtruth(object):
    """Use the minimum of the input and the groundtruth as actual groundtruth"""
    def __call__(self, sample):
        inputs = sample['laser'].clone()
        truths = sample['truth'].clone()
        min_truths = tc.min(inputs, truths)
        sample['truth'] = min_truths
        return sample

class AugmentRandomFlip(object):
    """Flip both input and groundtruth with a probability of p (default: p=0.5)"""
    def __init__(self, p=0.5):
        assert p >= 0 and p <=1
        self.p = p
        
    def __call__(self, sample):
        inputs = sample['laser'].clone()
        truths = sample['truth'].clone()
        inv_idx = tc.arange(inputs.size(1)-1, -1, -1).long()
        for row in range(inputs.size(0)):
            if (np.random.rand() < self.p):
                inputs[row] = inputs[row].index_select(0, inv_idx)
                truths[row] = truths[row].index_select(0, inv_idx)

        sample['laser'] = inputs
        sample['truth'] = truths
        return sample
    
class AugmentAddRandomUniformNoise(object):
    """Add random uniform noise in the range [-eps,eps) to the input"""
    def __init__(self, eps):
        assert isinstance(eps, float)
        self.eps = eps
        
    def __call__(self, sample):
        inputs = sample['laser'].clone()
        noise = (tc.rand(inputs.size()).double()-0.5) * 2 * self.eps
        sample['laser'] = inputs + noise
        return sample
    
class AugmentAddRandomGaussianNoise(object):
    """Add random Gaussian noise from N(mu, sigma) to the input"""
    def __init__(self, mu=0., sigma=1.):
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        self.mu = mu
        self.sigma = sigma
        
    def __call__(self, sample):
        inputs = sample['laser'].clone()
        noise = self.sigma * tc.randn(inputs.size()).double() + self.mu
        sample['laser'] = inputs + noise
        return sample

class ScaleToLength(object):
    """Subsample a 1D array until it fits the desired length"""
    def __init__(self, length):
        assert isinstance(length, int)
        self.length = length
    
    def __call__(self, sample):
        scaled = np.zeros(self.length)
        length = len(sample)
        ratio = length/float(self.length)
        lower = np.arange(0,length,ratio).astype(int)
        upper = np.hstack((lower[1:], [length]))
        for i in range(len(lower)):
            window = sample[lower[i]:upper[i]]
            if np.all(np.isnan(window)):
                m = np.nan
            else:
                m = np.nanmin(window)
            scaled[i] = m
        return scaled
            
class TrimToLength(object):
    """Trim equally the ends of a 1D array until it fits the desired length"""
    def __init__(self, length):
        assert isinstance(length, int)
        self.length = length
    
    def __call__(self, sample):
        length = len(sample)
        delta = length - self.length
        assert delta >= 0
        if delta > 0:
            sx = np.int(np.ceil(delta/2))
            dx = delta - sx
            sample = sample[sx:length-dx]
        
        return sample

class Interval(object):
    """Trim the ends of a 1D array to get the interval [sx:dx]"""
    def __init__(self, sx, dx):
        assert isinstance(sx, int)
        assert isinstance(dx, int)
        self.sx = sx
        self.dx = dx
    
    def __call__(self, sample):
        return sample[self.sx:self.dx]
        
class ReplaceInf(object):
    """Replace Inf values with the given number."""
    def __init__(self, num):
        assert isinstance(num, int)
        self.num = num
    
    def __call__(self, sample):
        mask = np.isinf(sample)
        sample[mask] = self.num
        return sample
    
class ReplaceNan(object):
    """Replace Nan values with the given number."""
    def __init__(self, num):
        assert isinstance(num, int)
        self.num = num
    
    def __call__(self, sample):
        mask = np.isnan(sample)
        sample[mask] = self.num
        return sample

class Normalize(object):
    """Normalize the sample bringing it in the range [0,1]."""
    def __init__(self, min_v, max_v):
        assert isinstance(min_v, (int, float))
        assert isinstance(max_v, (int, float))
        self.min_v = min_v
        self.max_v = max_v
    
    def __call__(self, sample):
        sample = (sample - self.min_v)/(self.max_v - self.min_v)
        return sample
    
class InvNormalize(object):
    """Reverse the effect of normalizing the sample in the range [0,1], bringing it back to [min_v,max_v]."""
    def __init__(self, min_v, max_v):
        assert isinstance(min_v, (int, float))
        assert isinstance(max_v, (int, float))
        self.min_v = min_v
        self.max_v = max_v
    
    def __call__(self, sample):
        sample = sample * (self.max_v - self.min_v) + self.min_v
        return sample
    
class NormalizeLog(object):
    """Normalize the sample bringing it in the range [0,1] by using a logarithmic scale."""
    def __init__(self, min_v, max_v):
        assert isinstance(min_v, (int, float))
        assert isinstance(max_v, (int, float))
        self.min_v = min_v
        self.max_v = max_v
    
    def __call__(self, sample):
        sample = np.log1p(sample - self.min_v)/np.log1p(self.max_v - self.min_v)
        return sample

class InvNormalizeLog(object):
    """Reverse the effect of normalizing the sample in the range [0,1] by using a logarithmic scale, bringing it back to [min_v,max_v]."""
    def __init__(self, min_v, max_v):
        assert isinstance(min_v, (int, float))
        assert isinstance(max_v, (int, float))
        self.min_v = min_v
        self.max_v = max_v
    
    def __call__(self, sample):
        sample = np.float_power(self.max_v - self.min_v + 1, sample) + self.min_v - 1
        return sample
        
class HallucinatingDataset(Dataset):
    """Hallucinating Robot dataset."""

    def __init__(self, csv_in_file, csv_out_file, transform_in=None, transform_out=None):
        """
        Args:
            csv_in_file (string): Path to the input csv file.
            csv_out_file (string): Path to the desired output (i.e. groundtruth) csv file.
            transform_in (callable, optional): Optional transform to be applied
                on a sample of the input set.
            transform_out (callable, optional): Optional transform to be applied
                on a sample of the output set.
        """
        self.laser_in = pd.read_csv(csv_in_file, delim_whitespace=True, header=None)
        self.truth_out = pd.read_csv(csv_out_file, delim_whitespace=True, header=None)
        self.transform_in = transform_in
        self.transform_out = transform_out
        self.length = len(self.laser_in)
        assert self.length == len(self.truth_out)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        laser = self.laser_in.loc[idx].values
        truth = self.truth_out.loc[idx].values
        if self.transform_in:
            laser = self.transform_in(laser)
            
        if self.transform_out:
            truth = self.transform_out(truth)
        sample = {'laser': laser, 'truth': truth}
        return sample