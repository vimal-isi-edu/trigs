# TRIGS: Trojan Identification from Gradient-based Signatures
This repository includes the official implementation for the following paper. The implementation is based on PyTorch.

Mohamed E. Hussein, Sudharshan Subramaniam Janakiraman, and Wael AbdAlmageed, "TRIGS: Trojan Identification from Gradient-based Signatures", accepted for ICPR 2024.

## Step 1: Prepare the datasets

### The ULP datasets for CIFAR10 and Tiny-ImageNet
Download the trained models for CIFAR10 and Tiny ImageNet from the following links
* CIFAR10: https://drive.google.com/drive/folders/1ye2KCRfzhbjtV3TMSRR5vlSBlvqNUqYL
* Tiny-ImageNet: https://drive.google.com/drive/folders/1shYf6mUn81p0ve1DQBFhxjE_B9JN1yKt

Unzip model files, which are named `clean_models_trainval.zip`, `poisoned_models_trainval.zip`, `clean_models_test.zip`, and `poisoned_models_test.zip`, for the CIFAR10 dataset, in a separate directory for each dataset. Note that for the Tiny ImageNet dataset, the `train` and `test` files for the poisoned models end with `Triggers_01_10` and `Triggers_11_20`, respectively, instead.

### The TAT dataset for ImageNet
Download the TAT dataset from [this repository](https://github.com/vimal-isi-edu/tat). Then, decompress all the `.tar.gz` files under the underlying four directories.

## Step 2: Set up the environment
This implementation was tested on an `Ubuntu 22.04` system with `Python 3.11.9` and the following packages, which you can also find in the included `requirements.txt` file.

```
torch 2.4.1
torchvision 0.19.1
lightning 2.4.0
wandb 0.18.1
matplotlib 3.9.2
pandas 2.2.2
scikit-learn 1.5.2
```

## Step 3: Create models' signatures
To create the signature for a probe model, use the script `trigs/generate_model_signature.py`. For example, for an ImageNet model from the TAT dataset, you can use the following command to create signatures with the same configurations used to produce the results in the paper. Note the `batch_size` can be adjusted based on the available memory on your `GPU`. The signature does not depend on the batch size.
```bash
python trigs/generate_model_signature.py \
  --dataset_name ImageNet \
  --model_name vitb16 \
  --weights_path PATH_TO_PYTORCH_MODEL_FILE \
  --iterations 200 \
  --learning_rate 0.1 \
  --output_dir OUTPUT_DIRECTORY \
  --opt_type ADAM \
  --lambda_tv 1e-3 \
  --batch_size 250
```

To learn about all the parameters and how to set them, please use the following command.
```bash
python trigs/generate_model_signature.py -h
```

The resulting signature is a set of _2N_ `.png` image files for an _N_-class model. For ImageNet, _N_=1000. Therefore, 2000 images will be created. For each class, there are two images, one for its activation minimization map and one for its activation maximization map.

The script `trigs/generate_model_signature.py` is designed to work with one model at a time. You will need to run it for all models in the dataset. The remaining scripts (below) assume the following directory structure for the resulting model signatures for the entire dataset of models.

```
<root directory of model signatures>
    clean
        train
            <directory of model signature image files 1>
            <directory of model signature image files 2>
            ...
        test
            <directory of model signature image files 1>
            <directory of model signature image files 2>
            ...
    poisoned
        train
            <directory of model signature image files 1>
            <directory of model signature image files 2>
            ...
        test
            <directory of model signature image files 1>
            <directory of model signature image files 2>
            ...
```

To create the statistics signatures from the raw image signatures, use the script `scripts/create_signature_stats_images.py` as follows. Note that the script runs only on the CPU, but it uses multiprocessing. `--num_jobs` should be set to the number of available CPU cores.

```bash
python scripts/create_signature_stats_images.py \
  --sig_path MODEL_SIGNATURE_DIRECTORY \
  --num_jobs NUMBER_OF_JOBS
```

The script above will create the following `numpy` files under each model signature directory in the dataset. The `numpy` files correspond to basic statistics (min, max, mean, and std), quantiles of different numbers (3 or 7), and histograms of different numbers of bins (4, 8, 12, or 16).
```
stats_basic.npy
stats_qls3.npy
stats_qls7.npy
stats_hist04.npy
stats_hist08.npy
stats_hist12.npy
stats_hist16.npy
```

## Step 4: Train and evaluate signature classifiers
After you create signatures for all the models in a dataset, following the directory structure above, you can train a signature classifier model using the script `trigs/signature_classifier.py`. For example, to train a classifier on the ImageNet dataset using the activation minimization maps, you can use the following.
```bash
python trigs/signature_classifier.py \
  --dataset_name ImageNet \
  --model_name Baseline \
  --epochs 200 \
  --ckpt_path OUTPUT_DIRECTORY \
  --data_path SIGNATURES_DIRECTORY \
  --opt_mode min \
  --split_rand_seed 10 \
  --train_ds_use_frac 1.0
```

To learn about all the parameters of the script and how to set them, including how to set the optimization mode, please use the following command.
```bash
python trigs/signature_classifier.py -h
```

Here are the seeds used to generate the different (up to 10) signature classification models described in the paper for each optimization mode.
```python
SPLIT_RAND_SEEDS = [43, 68, 12, 28, 22, 16, 85, 73, 95, 72]
```

## Citation
If you use this code, please cite the following paper.

```
@inproceedings{HusseinICPR24TRIGS,
  author       = {Mohamed E Hussein and
                  Sudharshan Subramaniam Janakiraman and
                  Wael AbdAlmageed},
  title        = {{TRIGS:} Trojan Identification from Gradient-based Signatures},
  booktitle    = {27th International Conference on Pattern Recognition, {ICPR} 2024},
  publisher    = {{Springer}},
  year         = {2024},
}
```