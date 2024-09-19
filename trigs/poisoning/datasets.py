from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import random
import os
from trigs.poisoning.utils import add_trigger_to_tensor


class TinyImageNet(ImageFolder):
    def __init__(self, root, split, **kwargs):
        self.split = split
        super(TinyImageNet, self).__init__(os.path.join(root, self.split), **kwargs)


class PoisonedDataset(Dataset):
    def __init__(self, trigger, target_class, poison_frac, dirty_label, dataset_class, **kwargs):
        self.dataset = dataset_class(**kwargs)
        self.trigger = trigger
        self.target_class = target_class
        self.poison_frac = poison_frac
        self.dirty_label = dirty_label
        self.targets = self.dataset.targets

    def __getitem__(self, index):
        sample, target = self.dataset[index]
        if random.random() < self.poison_frac:
            sample = add_trigger_to_tensor(sample, self.trigger)
            if self.dirty_label:
                target = self.target_class
        return sample, target

    def __len__(self):
        return len(self.dataset)
