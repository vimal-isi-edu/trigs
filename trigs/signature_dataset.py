# import Necessary Packages
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch


def absolute_file_paths(directory, opt_mode='all'):
    file_paths = []

    for folder, subs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.png'):
                if opt_mode == 'all' or \
                        (opt_mode == 'min' and filename[0] == '+') or \
                        (opt_mode == 'max' and filename[0] == '-'):
                    file_paths.append(os.path.abspath(os.path.join(folder, filename)))
    assert len(file_paths) > 0, "Model signature cannot be empty"
    return sorted(file_paths)


class SignaturePerImageDataset(Dataset):
    def __init__(self, dataset_folder: str = "", transform=None, opt_mode="all") -> None:
        super().__init__()

        self.images_path = absolute_file_paths(dataset_folder, opt_mode=opt_mode)
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img = Image.open(self.images_path[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


class SignatureDataset(Dataset):
    def __init__(self,
                 base_path: str, dataset_name: str, mode_: str = "train", model_type_: str = "clean",
                 baseline: bool = True, transform=None, opt_mode='all') -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.transform = transform
        self.mode_ = mode_
        self.baseline = baseline
        self.opt_mode = opt_mode
        if not self.opt_mode.startswith("stats"):
            if self.dataset_name == "CIFAR10":
                self.batch_size = 10
            elif self.dataset_name.startswith("TIN"):
                self.batch_size = 200
            elif self.dataset_name.startswith('ImageNet'):
                self.batch_size = 1000
            elif self.dataset_name == "TrojAI-Round01":
                self.batch_size = 5
            elif self.dataset_name in ["TrojAI-Round02", "TrojAI-Round03"]:
                self.batch_size = 25
            elif self.dataset_name == "TrojAI-Round04":
                self.batch_size = 45
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            if self.opt_mode == 'all':
                self.batch_size *= 2
        path = os.path.join(base_path, model_type_, mode_)
        model_list = [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
        self.path_list = sorted(model_list)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        label = 1.0 if self.path_list[index].__contains__(f"poisoned/{self.mode_}") else 0.0
        if self.opt_mode.startswith("stats"):
            stat_type = self.opt_mode.split('-')[1]
            stat_types = ['basic', f"qls{stat_type[1]}", f"hist{stat_type[-2:]}"]
            stats_data = []
            for st in stat_types:
                np_data = np.load(os.path.join(self.path_list[index], f"stats_{st}.npy")).astype(np.float32)
                stats_data.append(torch.as_tensor(np_data))
            img = torch.cat(stats_data)
        else:
            loader = DataLoader(SignaturePerImageDataset(self.path_list[index], self.transform, self.opt_mode),
                                shuffle=False, batch_size=self.batch_size)
            img = next(iter(loader))
            if len(img) < self.batch_size:
                padding = torch.zeros(size=(self.batch_size - len(img), *img.shape[1:]),
                                      dtype=img.dtype, device=img.device)
                img = torch.cat([img, padding])
            if self.baseline:
                img = img.view(img.shape[0] * img.shape[1], img.shape[2], img.shape[3])
        return img, label
