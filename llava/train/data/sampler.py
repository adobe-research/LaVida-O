import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
from itertools import chain
import random
import numpy as np
from torch.utils.data import DataLoader


class DataLoaderWithEpoch(DataLoader):

    def set_epoch(self, epoch: int):
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)
        if hasattr(self.batch_sampler, "sampler") and hasattr(self.batch_sampler.sampler, "set_epoch"):
            self.batch_sampler.sampler.set_epoch(epoch)
        if (
            hasattr(self.batch_sampler, "batch_sampler")
            and hasattr(self.batch_sampler.batch_sampler, "sampler")
            and hasattr(self.batch_sampler.batch_sampler.sampler, "set_epoch")
        ):
            self.batch_sampler.batch_sampler.sampler.set_epoch(epoch)
        # We support if a custom `Dataset` implementation has `set_epoch`
        # or in general HF datasets `Datasets`
        elif hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

class MultiDatasetBatchSampler(BatchSampler):
    def __init__(self, group_lengths, weights, batch_size,shuffle=True,local_rank=0,world_size=1,
                 seed=12345,group_bs_factor=None):
        self.datasets_length = np.array(group_lengths)
        self.datasets_start_index = np.cumsum( self.datasets_length)
        self.datasets_start_index = np.concatenate([[0],self.datasets_start_index])
        self.datasets_start_index,self.length = self.datasets_start_index[:-1],self.datasets_start_index[-1]
        self.dataset_weight = torch.tensor(weights).float()
        self.batch_size = batch_size
        self.local_rank = local_rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.group_bs_factor = group_bs_factor
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        print(f"Global Batch Size: {self.batch_size * self.world_size} ")
        print(f"Length: {self.length }")
    def set_seed(self,seed):
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device='cpu').manual_seed(seed)

    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def __iter__(self):
        batch = []
        self.set_seed(self.epoch)
        n_batches = self.length // self.batch_size
        for idx in range(n_batches):
            select_dataset = torch.multinomial(self.dataset_weight,1,replacement=True,generator=self.generator).item()
            selected_index = self.rng.integers(0,self.datasets_length[select_dataset],(self.world_size,self.batch_size))[self.local_rank]
            selected_index = selected_index + self.datasets_start_index[select_dataset]
            #print(f"RANK:{self.local_rank }",select_dataset)
            indices =  selected_index.tolist()
            bs_factor = 1
            if self.group_bs_factor is not None:
                bs_factor = self.group_bs_factor[select_dataset]
            old_bs = len(indices)
            new_bs = max(int(old_bs * bs_factor),1)
            # print(select_dataset,bs_factor,new_bs,old_bs)
            print(f"RANK:{self.local_rank } Groups {self.group_bs_factor} Selected {select_dataset}, BS_FAC {bs_factor}, NEW_BS {new_bs}, OLD_BS {old_bs}")
            if new_bs < old_bs:
                indices = indices[:new_bs]
            #print(f"Local Rank {self.local_rank} Batch ID {idx} : {indices}")
            yield indices

    def __len__(self):
        return self.length // (self.batch_size * self.world_size) 



if __name__ == '__main__':
    from torch.utils.data import TensorDataset
    dataset = TensorDataset(torch.rand(1000,3))
    sampler = MultiDatasetBatchSampler(
        [700,300],
        [1,1],
        2
    )
    dataloader = DataLoaderWithEpoch(dataset, batch_sampler=sampler)
    dataloader.set_epoch(0)
    print(0,next(iter(dataloader)))
    print(0,next(iter(dataloader)))
    dataloader.set_epoch(1)
    print(1,next(iter(dataloader)))
    dataloader.set_epoch(2)
    print(2,next(iter(dataloader)))
    dataloader.set_epoch(0)
    print(0,next(iter(dataloader)))
    print(0,next(iter(dataloader)))
    dataloader.set_epoch(1)
    print(1,next(iter(dataloader)))
    dataloader.set_epoch(2)
    print(2,next(iter(dataloader)))    
    breakpoint()