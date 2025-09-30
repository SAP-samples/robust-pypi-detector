from feature_extractor import FeatureExtractor
from feature_extractor_soa_detector import FeatureExtractorSoaDetector
import json
import os
from utils import Package
import pandas as pd
from train_ml_models import ML_MODELS, ML_MODELS_ADV, train_ml_models, advtrain_ml_models, \
    train_ml_models_full_dataset, advtrain_ml_models_full_dataset
import argparse
import random


NUM_FOLDS = 5

OUT_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/out'
DATA_BASE_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/data'
if not os.path.isdir(OUT_PATH):
    os.makedirs(OUT_PATH)
if not os.path.isdir(DATA_BASE_PATH):
    os.makedirs(DATA_BASE_PATH)

GOODWARE_PATH_PYPI_MALWAREBENCH = os.path.join(DATA_BASE_PATH, 'dataset_goodware_pypi_malwarebench')
MALWARE_PATH_PYPI_MALWAREBENCH = os.path.join(DATA_BASE_PATH, 'dataset_malware_pypi_malwarebench')
DATASET_PATH_PYPI_MALWAREBENCH = os.path.join(OUT_PATH, 'dataset_pypi_malwarebench.csv')
DATASET_PATH_PYPI_MALWAREBENCH_SOA_FEATURES = os.path.join(OUT_PATH, 'dataset_pypi_malwarebench_soa_features.csv')

MODELS_BASE_SAVE_PATH_PYPI_MALWAREBENCH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi')
MODELS_BASE_SAVE_PATH_PYPI_MALWAREBENCH_SOA_FEATURES = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_soa_features')
MODELS_BASE_SAVE_PATH_PYPI_MALWAREBENCH_FULL = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_full')
MODELS_BASE_SAVE_PATH_PYPI_MALWAREBENCH_FULL_SOA_FEATURES = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_full_soa_features')
# ADV TRAINING
DATASET_ADV_OUT_PATH_PYPI_MALWAREBENCH = os.path.join(OUT_PATH, 'datasets_advtrain_pypi_{model}_malwarebench')
ADV_PACKAGES_BASE_PATH_PYPI_MALWAREBENCH = os.path.join(DATA_BASE_PATH, '/home/mluser/data/dataset_adv_{model}_advtrain_malwarebench')
DATASET_ADV_OUT_PATH_PYPI_MALWAREBENCH_SOA_FEATURES = os.path.join(OUT_PATH, 'datasets_advtrain_pypi_{model}_malwarebench_soa_features')
ADV_PACKAGES_BASE_PATH_PYPI_MALWAREBENCH_SOA_FEATURES = os.path.join(DATA_BASE_PATH, 'dataset_adv_{model}_advtrain_malwarebench_soa_features')
# ADV_PACKAGES_BASE_PATH_PYPI = '/home/mluser/data/dataset_adv_{model}_advtrain_pypi'
MODELS_ADV_SAVE_PATH_PYPI_MALWAREBENCH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain')
MODELS_ADV_SAVE_PATH_PYPI_MALWAREBENCH_FULL = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain_full')
MODELS_ADV_SAVE_PATH_PYPI_MALWAREBENCH_SOA_FEATURES = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain_soa_features')
MODELS_ADV_SAVE_PATH_PYPI_MALWAREBENCH_FULL_SOA_FEATURES = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain_soa_features_full')

EVAL_RESULTS_PATH = os.path.join(OUT_PATH, 'eval_results_malwarebench/result_{model}_advtrain_fold{fold_idx}.json')
EVAL_RESULTS_PATH_NEW_DETECTOR = os.path.join(OUT_PATH, 'eval_results_malwarebench/result_{model}_advtrain_fold{fold_idx}_soa_features.json')
SAMPLES_PERC_ADV_TRAIN_DEFAULT = 0.2

ML_MODELS_FULL = ['xgboost']


def get_packages(base_path, pkg_type='pypi'):
    assert pkg_type in ['pypi', 'npm']
    packages = []

    for pkg in sorted([obj for obj in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, obj)) and not obj.startswith('.')]):
        pkg_name, pkg_version = pkg.rsplit('-', 1)
        packages.append(Package(pkg_name, pkg_version, pkg_type=pkg_type, path=os.path.join(base_path, pkg)))

    return packages


def build_dataset(goodware_path, malware_path, out_path, num_samples_per_class=None, ecosystem='pypi', use_soa_features=False):
    feat_extractor = FeatureExtractorSoaDetector() if use_soa_features else FeatureExtractor()

    benign_packages = get_packages(goodware_path, pkg_type=ecosystem)
    df_benign = feat_extractor.extract_features(benign_packages, label=0)

    malicious_packages = get_packages(malware_path, pkg_type=ecosystem)
    df_malware = feat_extractor.extract_features(malicious_packages, label=1)

    if num_samples_per_class is not None:
        if num_samples_per_class < len(df_benign.index) or num_samples_per_class < len(df_malware.index):
            num_samples_per_class = min(len(df_benign.index), len(df_malware.index))
        df_benign = df_benign.sample(num_samples_per_class, ignore_index=True, random_state=0)
        df_malware = df_malware.sample(num_samples_per_class, ignore_index=True, random_state=0)

    dataset = pd.concat([df_benign, df_malware])
    dataset.to_csv(out_path, index=False)


def build_dataset_adv(adv_packages_path, out_path, use_soa_features=False):
    feat_extractor = FeatureExtractorSoaDetector() if use_soa_features else FeatureExtractor()

    malicious_packages = get_packages(adv_packages_path)
    df_adv = feat_extractor.extract_features(malicious_packages, label=1)
    df_adv.to_csv(out_path, index=False)


def train_models_pypi_malwarebench(use_soa_features=False):
    dataset_path = DATASET_PATH_PYPI_MALWAREBENCH_SOA_FEATURES if use_soa_features else DATASET_PATH_PYPI_MALWAREBENCH
    if not os.path.isfile(dataset_path):
        print(f"Building dataset: {dataset_path}")
        build_dataset(GOODWARE_PATH_PYPI_MALWAREBENCH, MALWARE_PATH_PYPI_MALWAREBENCH, dataset_path, ecosystem='pypi', use_soa_features=use_soa_features)
    models_save_path = MODELS_BASE_SAVE_PATH_PYPI_MALWAREBENCH_SOA_FEATURES if use_soa_features else MODELS_BASE_SAVE_PATH_PYPI_MALWAREBENCH
    train_ml_models(ML_MODELS, dataset_path, NUM_FOLDS, models_save_path)


def adv_train_models_pypi_malwarebench(sample_perc=SAMPLES_PERC_ADV_TRAIN_DEFAULT, sort_samples=False, use_soa_features=False):
    assert sample_perc > 0.0 and sample_perc <= 1.0

    dataset_base_path = DATASET_PATH_PYPI_MALWAREBENCH_SOA_FEATURES if use_soa_features else DATASET_PATH_PYPI_MALWAREBENCH
    dataset_adv_out_base_path = DATASET_ADV_OUT_PATH_PYPI_MALWAREBENCH_SOA_FEATURES if use_soa_features else DATASET_ADV_OUT_PATH_PYPI_MALWAREBENCH
    adv_packages_base_path = ADV_PACKAGES_BASE_PATH_PYPI_MALWAREBENCH_SOA_FEATURES if use_soa_features else ADV_PACKAGES_BASE_PATH_PYPI_MALWAREBENCH
    models_save_path = MODELS_BASE_SAVE_PATH_PYPI_MALWAREBENCH_SOA_FEATURES if use_soa_features else MODELS_BASE_SAVE_PATH_PYPI_MALWAREBENCH
    models_adv_save_path = MODELS_ADV_SAVE_PATH_PYPI_MALWAREBENCH_SOA_FEATURES if use_soa_features else MODELS_ADV_SAVE_PATH_PYPI_MALWAREBENCH

    models_adv_save_path += f'_{int(sample_perc * 100)}'
    models_adv_save_path += "_sorted" if sort_samples else ""

    for model in ML_MODELS_ADV:
        adv_folder = dataset_adv_out_base_path.format(model=model)
        if not os.path.isdir(adv_folder):
            os.makedirs(adv_folder)
        if not os.path.isdir(adv_folder):
            os.makedirs(adv_folder)
        for fold_idx in range(1, NUM_FOLDS+1):
            out_path = os.path.join(adv_folder, f"dataset_advtrain_{model}_fold{fold_idx}.csv")
            adv_packages_path = os.path.join(adv_packages_base_path.format(model=model), f"fold{fold_idx}")
            if not os.path.isfile(out_path):
                build_dataset_adv(adv_packages_path, out_path, use_soa_features=use_soa_features)

    if sort_samples:
        eval_results_base_path = EVAL_RESULTS_PATH_NEW_DETECTOR if use_soa_features else EVAL_RESULTS_PATH
    else:
        eval_results_base_path = None
    advtrain_ml_models(ML_MODELS_ADV, dataset_base_path, dataset_adv_out_base_path, NUM_FOLDS, models_save_path, models_adv_save_path,
                       perc_adv_samples_fold=sample_perc, sort_by_score=sort_samples, eval_results_base_path=eval_results_base_path)


def train_models_pypi_full(use_soa_features=False):
    dataset_path = DATASET_PATH_PYPI_MALWAREBENCH if not use_soa_features else DATASET_PATH_PYPI_MALWAREBENCH_SOA_FEATURES
    models_base_savepath = MODELS_BASE_SAVE_PATH_PYPI_MALWAREBENCH_FULL if not use_soa_features else MODELS_BASE_SAVE_PATH_PYPI_MALWAREBENCH_FULL_SOA_FEATURES

    print(f"Running experiment for dataset: {dataset_path}")
    print(f"Saving models to: {models_base_savepath}")

    train_ml_models_full_dataset(ML_MODELS_FULL, dataset_path, models_base_savepath, log_results=True)


def adv_train_models_pypi_full(perc_adv_samples_fold=SAMPLES_PERC_ADV_TRAIN_DEFAULT, sort_samples=False, use_soa_features=False):
    dataset_path = DATASET_PATH_PYPI_MALWAREBENCH if not use_soa_features else DATASET_PATH_PYPI_MALWAREBENCH_SOA_FEATURES
    dataset_adv_out_base_path = DATASET_ADV_OUT_PATH_PYPI_MALWAREBENCH if not use_soa_features else DATASET_ADV_OUT_PATH_PYPI_MALWAREBENCH_SOA_FEATURES
    adv_packages_base_path = ADV_PACKAGES_BASE_PATH_PYPI_MALWAREBENCH if not use_soa_features else ADV_PACKAGES_BASE_PATH_PYPI_MALWAREBENCH_SOA_FEATURES
    models_adv_save_path = MODELS_ADV_SAVE_PATH_PYPI_MALWAREBENCH_FULL if not use_soa_features else MODELS_ADV_SAVE_PATH_PYPI_MALWAREBENCH_FULL_SOA_FEATURES

    for model in ML_MODELS_ADV:
        adv_folder = dataset_adv_out_base_path.format(model=model)
        if not os.path.isdir(adv_folder):
            os.makedirs(adv_folder)
        for fold_idx in range(1, NUM_FOLDS+1):
            out_path = os.path.join(adv_folder, f"dataset_advtrain_{model}_fold{fold_idx}.csv")
            adv_packages_path = os.path.join(adv_packages_base_path.format(model=model), f"fold{fold_idx}")
            if not os.path.isfile(out_path):
                build_dataset_adv(adv_packages_path, out_path)

    perc_str = f'_{int(perc_adv_samples_fold * 100)}' if perc_adv_samples_fold is not None else '_100'
    models_adv_save_path += perc_str
    models_adv_save_path += "_sorted" if sort_samples else ""
    if sort_samples:
        eval_results_base_path = EVAL_RESULTS_PATH if not use_soa_features else EVAL_RESULTS_PATH_NEW_DETECTOR
    else:
        eval_results_base_path = None
    advtrain_ml_models_full_dataset(ML_MODELS_ADV, dataset_path, dataset_adv_out_base_path, models_adv_save_path,
                                    perc_adv_samples_fold=perc_adv_samples_fold, rand_state=0,
                                    sort_by_score=sort_samples, eval_results_base_path=eval_results_base_path, log_results=True)


def adv_train_models_pypi_full_perc(sort_samples=False, use_soa_features=False):
    for perc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        adv_train_models_pypi_full(perc_adv_samples_fold=perc, sort_samples=sort_samples, use_soa_features=use_soa_features)


def train_final_detector():
    datasets = [
        ("base", os.path.join(OUT_PATH, "dataset_pypi_malwarebench_full.csv"), os.path.join(OUT_PATH, "dataset_pypi_malwarebench.csv"), os.path.join(OUT_PATH, "dataset_pypi_live_final_clean.csv")),
        ("soa", os.path.join(OUT_PATH, "dataset_pypi_malwarebench_soa_features_full.csv"), os.path.join(OUT_PATH, "dataset_pypi_malwarebench_soa_features.csv"), os.path.join(OUT_PATH, "dataset_pypi_live_soa_features.csv")),
    ]

    for dataset_type, dataset_full_path, dataset_base_path, dataset_live_path in datasets:
        print(f"Running experiment for dataset: {dataset_full_path}")
        if not os.path.isfile(dataset_full_path):
            dataset_base = pd.read_csv(dataset_base_path)
            dataset_live = pd.read_csv(dataset_live_path)
            assert len(dataset_base.columns) == len(dataset_live.columns)
            dataset_full = pd.concat([dataset_base, dataset_live], ignore_index=True)
            assert len(dataset_full.index) == len(dataset_base.index) + len(dataset_live.index)
            # check that there are no NaN values
            assert dataset_full.isnull().values.any() == False
            # remove duplicated rows (if any)
            dataset_full.drop_duplicates(inplace=True)
            dataset_full.to_csv(dataset_full_path, index=False)

        if dataset_type == "base":
            models_save_path = MODELS_BASE_SAVE_PATH_PYPI_MALWAREBENCH + "_definitive"
        else:
            models_save_path = MODELS_BASE_SAVE_PATH_PYPI_MALWAREBENCH_SOA_FEATURES + f"_definitive"
        train_ml_models(ML_MODELS_FULL, dataset_full_path, NUM_FOLDS, models_save_path)


def train_final_detector_full():
    datasets = [
        ("base", os.path.join(OUT_PATH, "dataset_pypi_malwarebench_full.csv"), os.path.join(OUT_PATH, "dataset_pypi_malwarebench.csv"), os.path.join(OUT_PATH, "dataset_pypi_live_full.csv")),
        ("soa", os.path.join(OUT_PATH, "dataset_pypi_malwarebench_soa_features_full.csv"), os.path.join(OUT_PATH, "dataset_pypi_malwarebench_soa_features.csv"), os.path.join(OUT_PATH, "dataset_pypi_live_full_soa_features.csv")),
    ]

    for dataset_type, dataset_full_path, dataset_base_path, dataset_live_path in datasets:
        print(f"Running experiment for dataset: {dataset_full_path}")
        if not os.path.isfile(dataset_full_path):
            dataset_base = pd.read_csv(dataset_base_path)
            dataset_live = pd.read_csv(dataset_live_path)
            assert len(dataset_base.columns) == len(dataset_live.columns)
            dataset_full = pd.concat([dataset_base, dataset_live], ignore_index=True)
            assert len(dataset_full.index) == len(dataset_base.index) + len(dataset_live.index)
            # check that there are no NaN values
            assert dataset_full.isnull().values.any() == False
            # remove duplicated rows (if any)
            dataset_full.drop_duplicates(inplace=True)
            dataset_full.to_csv(dataset_full_path, index=False)

        if dataset_type == "base":
            models_save_path = '/home/mluser/workspace/training_results_malwarebench/training_results_pypi_definitive_full'
        else:
            models_save_path = f'/home/mluser/workspace/training_results_malwarebench/training_results_pypi_soa_features_definitive_full'
        train_ml_models_full_dataset(ML_MODELS_FULL, dataset_full_path, models_save_path, log_results=True)


def adv_train_final_detector_full(perc_adv_samples_fold=0.2, sort_samples=True):
    datasets = [
        ("base", os.path.join(OUT_PATH, "dataset_pypi_malwarebench_full.csv")),
        ("new", os.path.join(OUT_PATH, "dataset_pypi_malwarebench_soa_features_full.csv")),
    ]

    for dataset_type, dataset_full_path in datasets:
        dataset_path = dataset_full_path
        dataset_adv_out_base_path = DATASET_ADV_OUT_PATH_PYPI_MALWAREBENCH if dataset_type == "base" else DATASET_ADV_OUT_PATH_PYPI_MALWAREBENCH_SOA_FEATURES
        adv_packages_base_path = ADV_PACKAGES_BASE_PATH_PYPI_MALWAREBENCH if dataset_type == "base" else ADV_PACKAGES_BASE_PATH_PYPI_MALWAREBENCH_SOA_FEATURES
        models_adv_save_path = MODELS_ADV_SAVE_PATH_PYPI_MALWAREBENCH_FULL if dataset_type == "base" else MODELS_ADV_SAVE_PATH_PYPI_MALWAREBENCH_FULL_SOA_FEATURES

        models_adv_save_path = models_adv_save_path.replace("_full", "_definitive_full")
        models_adv_save_path += "_20_sorted"
        if sort_samples:
            eval_results_base_path = EVAL_RESULTS_PATH if dataset_type == "base" else EVAL_RESULTS_PATH_NEW_DETECTOR
        else:
            eval_results_base_path = None
        advtrain_ml_models_full_dataset(ML_MODELS_ADV, dataset_path, dataset_adv_out_base_path, models_adv_save_path,
                                        perc_adv_samples_fold=perc_adv_samples_fold, rand_state=0,
                                        sort_by_score=sort_samples, eval_results_base_path=eval_results_base_path, log_results=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train malicious packages detectors.')
    parser.add_argument('mode', type=str, default="train-base", help='Type of experiment to run (train-base, adv-train, ...)')
    parser.add_argument('--soa-features', action='store_true', help='Use SoA features')
    args = parser.parse_args()

    print(f"Running experiment: {args.mode}")

    if args.mode == 'train-base':
        train_models_pypi_malwarebench(use_soa_features=args.soa_features)
    elif args.mode == 'train-full':
        train_models_pypi_full(use_soa_features=args.soa_features)
    elif args.mode == 'adv-train':
        adv_train_models_pypi_malwarebench(sample_perc=0.2, sort_samples=True, use_soa_features=args.soa_features)
    elif args.mode == 'adv-train-full':
        adv_train_models_pypi_full(perc_adv_samples_fold=SAMPLES_PERC_ADV_TRAIN_DEFAULT, sort_samples=True, use_soa_features=args.soa_features)
    elif args.mode == 'adv-train-full-perc':
        adv_train_models_pypi_full_perc(sort_samples=True, use_soa_features=args.soa_features)
    elif args.mode == 'train-final-detector':
        train_final_detector()
    elif args.mode == 'train-final-detector-full':
        train_final_detector_full()
    elif args.mode == 'adv-train-final-detector-full':
        adv_train_final_detector_full(sort_samples=True)
