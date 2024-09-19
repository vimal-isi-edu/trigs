import os
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.transforms import (
    Normalize, ToTensor, Compose, RandomCrop, CenterCrop, RandomHorizontalFlip, ColorJitter
)
from torchvision.datasets import ImageNet
from PIL import Image
from datetime import datetime
from trigs.poisoning.datasets import PoisonedDataset, TinyImageNet
from lightning.pytorch import LightningModule, LightningDataModule, Trainer, loggers, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from trigs.poisoning.utils import create_trigger
from typing import Callable, Type
import multiprocessing as mp
import logging
from torchmetrics.classification import MulticlassConfusionMatrix
from trigs.custom_models import VGG16TIN, resnet18_mod
from trigs.utils import deterministic_subset_label_distribution, random_split_label_distribution


class PoisonedDataModule(LightningDataModule):
    def __init__(self,
                 dataset_class: Type[Dataset],
                 trigger: torch.Tensor,
                 train_poison_frac: float,
                 target_class: int,
                 train_transform: Callable,
                 eval_transform: Callable,
                 data_path: [None, str] = None,
                 batch_size: int = 8,
                 num_workers: int = 2,
                 data_ready=None,
                 process_id: int = 0,
                 subset_start: float = 0.0,
                 subset_end: float = 1.0,
                 seed: int = 43):
        super().__init__()
        self.dataset_class = dataset_class
        self.trigger = trigger
        self.train_poison_frac = train_poison_frac
        self.target_class = target_class
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.data_path = data_path or os.getenv('TMPDIR')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_ready = data_ready
        self.process_id = process_id
        self.subset_start = subset_start
        self.subset_end = subset_end
        self.seed = seed

        # initialize other member variables
        self.train_ds = self.train_dl = None
        self.valid_ds = self.valid_dl = None
        self.clean_test_ds = self.clean_test_dl = None
        self.poisoned_test_ds = self.poisoned_test_dl = None
        self.test_clean = True

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        if self.process_id > 0 and self.data_ready is not None:
            self.data_ready.wait()
        # create the training and validation datasets
        train_ds = PoisonedDataset(self.trigger, self.target_class, self.train_poison_frac, dirty_label=True,
                                   dataset_class=self.dataset_class,
                                   root=self.data_path, split='train', transform=self.train_transform)
        valid_ds = PoisonedDataset(self.trigger, self.target_class, self.train_poison_frac, dirty_label=True,
                                   dataset_class=self.dataset_class,
                                   root=self.data_path, split='train', transform=self.eval_transform)
        labels = train_ds.targets  # If targets are not available, we will have to loop over the dataset object to
        # collect the labels. In this case, we must disable poisoning or disable dirty_label to get clean targets.
        subset_indices = deterministic_subset_label_distribution(labels, self.subset_start, self.subset_end)
        if subset_indices is not None:
            train_ds = torch.utils.data.Subset(train_ds, subset_indices)
            valid_ds = torch.utils.data.Subset(valid_ds, subset_indices)
            labels = [labels[idx] for idx in subset_indices]
        train_indices, val_indices = random_split_label_distribution(labels, (0.9, 0.1), random_state=self.seed)
        self.train_ds = torch.utils.data.Subset(train_ds, train_indices)
        self.valid_ds = torch.utils.data.Subset(valid_ds, val_indices)

        # create the testing datasets
        self.clean_test_ds = PoisonedDataset(self.trigger, self.target_class, 0.0, dirty_label=False,
                                             dataset_class=self.dataset_class,
                                             root=self.data_path, split='val', transform=self.eval_transform)
        self.poisoned_test_ds = PoisonedDataset(self.trigger, self.target_class, 1.0, dirty_label=False,
                                                dataset_class=self.dataset_class,
                                                root=self.data_path, split='val', transform=self.eval_transform)
        if self.process_id == 0 and self.data_ready is not None:
            self.data_ready.set()

    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                                   num_workers=self.num_workers, persistent_workers=True)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False,
                                   num_workers=self.num_workers, persistent_workers=True)
        self.clean_test_dl = DataLoader(self.clean_test_ds, batch_size=self.batch_size, shuffle=False,
                                        num_workers=self.num_workers, persistent_workers=True)
        self.poisoned_test_dl = DataLoader(self.poisoned_test_ds, batch_size=self.batch_size, shuffle=False,
                                           num_workers=self.num_workers, persistent_workers=True)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.valid_dl

    def test_dataloader(self):
        if self.test_clean:
            return self.clean_test_dl
        return self.poisoned_test_dl


class LightningModelWrapper(LightningModule):
    def __init__(self, model: nn.Module, num_classes: int,
                 optim_class: Type[torch.optim.Optimizer], optim_config: dict) -> None:
        super().__init__()

        self.model = model
        self.optim_class = optim_class
        self.optim_config = optim_config
        self.test_cm = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true")
        self.test_step_outputs = []
        self.val_step_outputs = []

    def forward(self, batch):
        return self.model(batch[0])

    def configure_optimizers(self):
        optimizer = self.optim_class(self.parameters(), **self.optim_config)
        return optimizer

    def test_step(self, test_batch, test_batch_idx):
        x, y = test_batch
        z = self.model(x)
        n_correct = torch.sum(torch.argmax(z, dim=1).eq(y)).item()
        n_samples = len(y)
        self.test_cm(z, y)
        self.test_step_outputs.append([n_correct, n_samples])
        return n_correct, n_samples

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        tot_correct = sum([out[0] for out in outputs])
        tot_samples = sum([out[1] for out in outputs])
        self.log("Test Accuracy", tot_correct / tot_samples)
        cm = self.test_cm.compute()
        wandb_logger = self.logger.experiment
        wandb_logger.log({"Test CM": wandb.Image(cm, caption="Confusion Matrix")})
        self.test_cm.reset()
        self.test_step_outputs.clear()

    def compute_loss_(self, batch):
        x, y = batch
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        return loss, z, y

    def training_step(self, train_batch, train_batch_idx):
        loss, z, y = self.compute_loss_(train_batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        n_correct = torch.sum(torch.argmax(z, dim=1).eq(y)).item()
        n_samples = len(y)
        self.log("train_acc", n_correct / n_samples, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        loss, z, y = self.compute_loss_(val_batch)
        n_correct = torch.sum(torch.argmax(z, dim=1).eq(y)).item()
        n_samples = len(y)
        self.log("val_loss_step", loss)
        acc = n_correct / n_samples
        self.log("val_acc_step", acc)
        self.val_step_outputs.append([loss, acc])
        return loss, acc

    def on_validation_epoch_end(self):
        outputs = self.val_step_outputs
        loss = torch.mean(torch.as_tensor([out[0] for out in outputs]))
        acc = torch.mean(torch.as_tensor([out[1] for out in outputs]))
        self.log("val_loss_epoch", loss)
        self.log("val_acc_epoch", acc, on_epoch=True, prog_bar=True, logger=True)
        self.val_step_outputs.clear()


def launch_one_experiment(args, exp_name, device_id, data_ready=None):
    starting_time = datetime.now()

    out_dir = args.out_dir
    data_dir = args.data_dir
    batch_size = args.batch_size
    epochs = args.epochs
    num_workers = args.num_workers
    poison_frac = args.poison_frac
    dataset = args.dataset
    subset_start = args.subset_start
    subset_end = args.subset_end
    tin_model_name = args.tin_model_name

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if dataset == "ImageNet":
        dataset_class = ImageNet
        num_classes = 1000
        trigger_size = 32
        train_transform = eval_transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        optim_class = torch.optim.AdamW
        optim_config = {"lr": 1e-5}
    elif dataset == "TIN":
        dataset_class = TinyImageNet
        num_classes = 200
        trigger_size = 8
        train_transform = Compose([
            RandomCrop(56),
            RandomHorizontalFlip(),
            ColorJitter(saturation=(0.5, 2.0)),  # from https://learningai.io/projects/2017/06/29/tiny-imagenet.html
            ToTensor(),
            normalize,
        ])
        eval_transform = Compose([
            CenterCrop(56),
            ToTensor(),
            normalize,
        ])
        if tin_model_name == "resnet10":
            # The following are the config for the resnet18_mod model
            optim_class = torch.optim.Adam
            optim_config = {'lr': 0.001}
        elif tin_model_name == "vgg16":
            # The following are the config for the VGG16TIN model
            optim_class = torch.optim.SGD
            optim_config = {'lr': 0.01, 'momentum': 0.9, 'nesterov': True, "weight_decay": 1e-3}
        else:
            raise ValueError(f"Unsupported model_name for the TIN dataset: {tin_model_name}")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    trigger_transform = Compose([ToTensor(), normalize])

    # create experiment name and checkpoint directory
    ckpt_dir = str(os.path.join(out_dir, exp_name))
    os.makedirs(ckpt_dir, exist_ok=True)

    # create a logger for this process
    log_formatter = logging.Formatter(f"%(asctime)s [device: {device_id}] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()

    file_handler = logging.FileHandler(os.path.join(ckpt_dir, f"{exp_name}.log"))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)

    # load or create trigger and transform it to a tensor
    if poison_frac == 0:
        trigger = None
    else:
        trigger_file_path = os.path.join(ckpt_dir, 'trigger.png')
        if os.path.exists(trigger_file_path):
            trigger = Image.open(trigger_file_path)
            trigger.load()
        else:
            trigger = create_trigger(base_size=5, target_size=trigger_size)
            trigger.save(trigger_file_path)
        trigger = trigger_transform(trigger)

    # load or choose random seed and target class
    md_file_path = os.path.join(ckpt_dir, 'metadata.json')
    if os.path.exists(md_file_path):
        with open(md_file_path, 'rt') as fp:
            metadata = json.load(fp)
        target_class = metadata.get('target_class')
        seed = metadata['seed']
    else:
        seed = random.randrange(0, int(1e9))
        metadata = {'seed': seed}
        if poison_frac == 0:
            target_class = None
        else:
            target_class = random.randrange(0, num_classes)
            metadata['target_class'] = target_class
        with open(md_file_path, 'wt') as fp:
            json.dump(metadata, fp, indent=4)

    seed_everything(seed, workers=True)

    # now, initialize models after setting the random seed
    if dataset == "ImageNet":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    elif dataset == "TIN":
        if tin_model_name == "resnet10":
            model = resnet18_mod(num_classes=num_classes)
        else:
            # The following are the config for the VGG16TIN model
            # Use Kaiming He Normal init as described in https://learningai.io/projects/2017/06/29/tiny-imagenet.html
            model = VGG16TIN()
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # wrap model in a lightning module
    pl_module = LightningModelWrapper(model, num_classes=num_classes,
                                      optim_class=optim_class, optim_config=optim_config)

    # create data module
    pl_data_module = PoisonedDataModule(trigger=trigger, train_poison_frac=poison_frac, target_class=target_class,
                                        train_transform=train_transform, eval_transform=eval_transform,
                                        data_path=data_dir, batch_size=batch_size, num_workers=num_workers,
                                        data_ready=data_ready, process_id=device_id, dataset_class=dataset_class,
                                        subset_start=subset_start, subset_end=subset_end, seed=seed)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=True,
        every_n_epochs=1,
        monitor="val_loss_epoch",
        mode="min",
        dirpath=ckpt_dir,
    )

    wandb_logger = loggers.WandbLogger(
        project=f"TRIGS",
        name=f"{exp_name}",
        id=f"{exp_name}",
        config=args.__dict__,
        save_dir=os.path.join("wandb", exp_name),
        resume="allow",
    )

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        devices=[device_id] if torch.cuda.is_available() else 'auto',
    )

    # test the original model before training
    logger.info("Testing the initial model:")
    logger.info("==========================")
    pl_data_module.test_clean = True
    trainer.test(model=pl_module, datamodule=pl_data_module)

    # train or resume training
    logger.info("Training/Fine-tuning the model on poisoned data:")
    logger.info("================================================")
    last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    trainer.fit(pl_module, pl_data_module, ckpt_path=last_ckpt_path if os.path.exists(last_ckpt_path) else None)

    test_results = dict()
    if poison_frac > 0.0:
        # test on poisoned data only if the model was poisoned
        logger.info("Testing on POISONED data:")
        logger.info("=========================")
        pl_data_module.test_clean = False
        ptest_res = trainer.test(dataloaders=pl_data_module.test_dataloader(), ckpt_path='best')
        test_results |= {"Poisoned " + k: v for k, v in ptest_res[0].items()}
        pred = trainer.predict(dataloaders=pl_data_module.test_dataloader(), ckpt_path='best')
        all_logits = torch.cat(pred)
        attack_success_rate = torch.sum(torch.argmax(all_logits, dim=1) == target_class) / len(all_logits)
        logger.info(f"Attack Success Rate = {attack_success_rate}")
        test_results["Attack Success Rate"] = attack_success_rate.item()

    # test on clean data
    logger.info("Testing on CLEAN data:")
    logger.info("======================")
    pl_data_module.test_clean = True
    ctest_res = trainer.test(dataloaders=pl_data_module.test_dataloader(), ckpt_path='best')
    test_results |= {"Clean " + k: v for k, v in ctest_res[0].items()}

    with open(os.path.join(ckpt_dir, 'test_results.json'), "wt") as fp:
        json.dump(test_results, fp, indent=4)

    logger.info(f"ELAPSED TIME IS {datetime.now() - starting_time}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_names", nargs="+", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--poison_frac", type=float, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--tin_model_name", type=str,
                        help="Name of the model architecture to be used in the case of the TIN dataset")
    parser.add_argument("--subset_start", type=float, default=0.0,
                        help="Beginning fraction of the subset of the training dataset used for training")
    parser.add_argument("--subset_end", type=float, default=1.0,
                        help="Ending fraction of the subset of the training dataset used for training")

    # parse arguments
    args = parser.parse_args()

    # print device name before going into the parallel or sequential execution
    if torch.cuda.is_available():
        print(f"CUDA DEVICE: {torch.cuda.get_device_name()}")

    # # sequential execution of experiments
    # for i, exp_name in enumerate(args.exp_names):
    #     launch_one_experiment(args, exp_name, i)

    # parallel execution of experiments
    mp.set_start_method('spawn')  # this is necessary to be able to initialize CUDA
    with mp.Manager() as manager:
        # using a process manager is necessary to be able to share the Event object among processes
        data_ready = manager.Event()
        # we cannot use a Pool executer here because pool processes are daemons, and they do not support child processes
        # which are needed by lightning or pytorch I guess
        # create processes
        processes = [mp.Process(target=launch_one_experiment,
                                args=(args, exp_name, i, data_ready)) for i, exp_name in enumerate(args.exp_names)]
        # start all processes
        for p in processes:
            p.start()
        # wait for all to finish before clearing the process manager and exiting the main thread
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
