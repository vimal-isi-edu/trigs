import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from joblib import parallel_backend, Parallel, delayed


def verify_histogram(histogram):
    assert np.all(np.abs(histogram.sum(axis=0) - 1) < 1e-6)


def half_histogram_bins(histogram):
    nbins = len(histogram)
    assert nbins % 2 == 0, "num bins must be even to be able to half histogram bins"
    half_hist = np.empty(shape=(histogram.shape[0] // 2, *histogram.shape[1:]), dtype=histogram.dtype)
    for b in range(len(half_hist)):
        half_hist[b] = histogram[2 * b] + histogram[2 * b + 1]
    verify_histogram(half_hist)
    return half_hist


def compute_histogram(imgs, nbins):
    edges = np.linspace(0, 1, nbins + 1)
    edges[-1] += 1e-6  # to make the check for the ending edge inclusive
    bins = []
    for i in range(len(edges) - 1):
        in_bin = np.logical_and(imgs >= edges[i], imgs < edges[i + 1])
        bin_vals = np.sum(in_bin, axis=2).astype(np.float32)
        bins.append(bin_vals)
    histogram = np.stack(bins) / imgs.shape[2]
    verify_histogram(histogram)
    return histogram


def compute_histograms(imgs):
    hist16 = compute_histogram(imgs, 16)
    hist12 = compute_histogram(imgs, 12)
    hist8 = half_histogram_bins(hist16)
    hist4 = half_histogram_bins(hist8)
    return hist4, hist8, hist12, hist16


def compute_quantiles(imgs):
    imgs_7qls = np.quantile(imgs, q=[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], axis=2)
    imgs_3qls = imgs_7qls[[1, 3, 5], ...]
    return imgs_3qls, imgs_7qls


def compute_basic_stats(imgs):
    imgs_min = np.min(imgs, axis=2)
    imgs_max = np.max(imgs, axis=2)
    imgs_avg = np.mean(imgs, axis=2)
    imgs_std = np.std(imgs, axis=2)
    return np.stack([imgs_min, imgs_max, imgs_avg, imgs_std])


stat_names = [
        'basic',
        'qls3',
        'qls7',
        'hist04',
        'hist08',
        'hist12',
        'hist16',
    ]


def compute_stats(imgs):
    basic_stats = compute_basic_stats(imgs)
    quantiles = compute_quantiles(imgs)
    histograms = compute_histograms(imgs)
    return {
        'basic': basic_stats,
        'qls3': quantiles[0],
        'qls7': quantiles[1],
        'hist04': histograms[0],
        'hist08': histograms[1],
        'hist12': histograms[2],
        'hist16': histograms[3],
    }


def process_one_dir(sd):
    if all(os.path.exists(os.path.join(sd, f"stats_{sname}.npy")) for sname in stat_names):
        return
    min_sig_img_files = glob.glob(os.path.join(sd, "+*.png"))
    min_sig_imgs = np.concatenate([np.asarray(Image.open(img_file)) for img_file in min_sig_img_files], axis=2)
    min_sig_imgs = np.float32(min_sig_imgs) / 255
    min_stats = compute_stats(min_sig_imgs)
    max_sig_img_files = glob.glob(os.path.join(sd, "-*.png"))
    max_sig_imgs = np.concatenate([np.asarray(Image.open(img_file)) for img_file in max_sig_img_files], axis=2)
    max_sig_imgs = np.float32(max_sig_imgs) / 255
    max_stats = compute_stats(max_sig_imgs)
    all_stats = compute_stats(np.concatenate([min_sig_imgs, max_sig_imgs], axis=2))
    for sname, all_stat_val in all_stats.items():
        np.save(os.path.join(sd, f"stats_{sname}.npy"),
                np.concatenate([min_stats[sname], max_stats[sname], all_stat_val]))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sig_path", type=str, required=True,
                        help='Path to created model signatures for all models.'
                             ' Expected structure is [poisoned|clean]/[train|test]/<model_number>/.')
    parser.add_argument("--num_jobs", type=int, default=1,
                        help='Number of parallel jobs. Should match the number of available CPU cores.')
    args = parser.parse_args()
    sig_path = args.sig_path
    num_jobs = args.num_jobs
    print(sig_path)
    sig_dirs = glob.glob(os.path.join(sig_path, "clean/*/*/")) + glob.glob(os.path.join(sig_path, "poisoned/*/*/"))
    if num_jobs > 1:
        with parallel_backend('loky', n_jobs=num_jobs):
            Parallel()(delayed(process_one_dir)(sd) for sd in tqdm(sig_dirs))
    else:
        for sd in tqdm(sig_dirs):
            process_one_dir(sd)


if __name__ == "__main__":
    main()
