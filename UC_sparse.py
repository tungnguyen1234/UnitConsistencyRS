import torch as t
import gc
import torch as t
from torch.sparse import FloatTensor
import pynvml
import psutil

# pynvml.nvmlInit()

def print_memory_usage():
    process = psutil.Process()
    print(f"CPU Memory: {process.memory_info().rss / 1e9:.2f} GB")
    
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU Memory: {info.used / 1e9:.2f} GB / {info.total / 1e9:.2f} GB")

def gpu_usage():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

device = t.device(f"cuda:0" if t.cuda.is_available() else "cpu")
epsilon = 1e-9

class SparseUC:
    def __init__(self, device, tensor, epsilon, mode_full=True):
        self.device = device
        if tensor.is_sparse:
            self.tensor = tensor.coalesce().to(device)
        else:
            self.tensor = tensor.to_sparse().coalesce().to(device)
        self.epsilon = epsilon
        self.mode_full=  mode_full

    def UC(self):
        indices = self.tensor.indices()
        values = self.tensor.values().log()
        d1, d2 = self.tensor.size()

        sigma_first = t.bincount(indices[0], minlength=d1).to(self.device)
        sigma_second = t.bincount(indices[1], minlength=d2).to(self.device)

        latent_1 = t.zeros(d1, device=self.device)
        latent_2 = t.zeros(d2, device=self.device)

        errors = []
        step = 1
        while True:
            if step % 2 == 0:
                rho_second = -t.zeros(d2, device=self.device).index_add_(0, indices[1], values).div_(sigma_second).nan_to_num(0.0)
                values.add_(rho_second[indices[1]])
                latent_2.sub_(rho_second)
                error = rho_second.pow(2).sum()

                rho_first = -t.zeros(d1, device=self.device).index_add_(0, indices[0], values).div_(sigma_first).nan_to_num(0.0)
                values.add_(rho_first[indices[0]])
                latent_1.sub_(rho_first)
                error += rho_first.pow(2).sum()
            else:
                rho_first = -t.zeros(d1, device=self.device).index_add_(0, indices[0], values).div_(sigma_first).nan_to_num(0.0)
                values.add_(rho_first[indices[0]])
                latent_1.sub_(rho_first)
                error = rho_first.pow(2).sum()

                rho_second = -t.zeros(d2, device=self.device).index_add_(0, indices[1], values).div_(sigma_second).nan_to_num(0.0)
                values.add_(rho_second[indices[1]])
                latent_2.sub_(rho_second)
                error += rho_second.pow(2).sum()

            if step % 5 == 0:  # Print every 10 steps
                print(f"Step {step}:")
                # print_memory_usage()
                # print(f"GPU Utilization: {gpu_usage()}%")
                print(f"Error: {error}")
                print()
            step += 1
            if error < self.epsilon:
                break
        
        if self.mode_full:
            values = FloatTensor(indices, values, self.tensor.size()).to_dense()
            tensor_return = t.exp(latent_1[:, None] + values + latent_2[None, :])
            return tensor_return
    
        else:
            return t.exp(latent_1), t.exp(latent_2)