import argparse
from feature_extractor import FeatureExtractor
from feature_extractor_soa_detector import FeatureExtractorSoaDetector
from run_training import get_packages
import joblib
import json
import os
from utils import Package
import pandas as pd
import requests
import re
import multiprocessing
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/out'
DATA_BASE_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/data'


ML_MODEL_DEFAULT_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_full/xgboost_final.joblib')
ADVTRAIN_MODEL_DEFAULT_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain_full_20_sorted/xgboost_final_20.joblib')
ADVTRAIN_MODEL_PERC_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain_full_{perc}/xgboost_final_{perc}.joblib')
ADVTRAIN_MODEL_PERC_PATH_SORTED = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain_full_{perc}_sorted/xgboost_final_{perc}.joblib')
ML_MODEL_DEFINITIVE_DEFAULT_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_definitive_full/xgboost_final.joblib')
ADVTRAIN_MODEL_DEFINITIVE_DEFAULT_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain_definitive_full_20_sorted/xgboost_final_20.joblib')

ML_MODEL_SOA_FEATURES_DEFAULT_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_full_soa_features/xgboost_final.joblib')
ADVTRAIN_MODEL_SOA_FEATURES_DEFAULT_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain_soa_features_full_20_sorted/xgboost_final_20.joblib')
ADVTRAIN_MODEL_SOA_FEATURES_PERC_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain_soa_features_full_{perc}/xgboost_final_{perc}.joblib')
ADVTRAIN_MODEL_SOA_FEATURES_PERC_PATH_SORTED = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain_soa_features_full_{perc}_sorted/xgboost_final_{perc}.joblib')
ML_MODEL_SOA_FEATURES_DEFINITIVE_DEFAULT_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_soa_features_definitive_full/xgboost_final.joblib')
ADVTRAIN_MODEL_SOA_FEATURES_DEFINITIVE_DEFAULT_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain_soa_features_definitive_full_20_sorted/xgboost_final_20.joblib')

DATASET_PATH = os.path.join(OUT_PATH, 'dataset_pypi_malwarebench.csv')
DATASET_SOA_FEATURES_PATH = os.path.join(OUT_PATH, 'dataset_pypi_malwarebench_soa_features.csv')

TARGET_PERC = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

RESULTS_BASE_PATH = os.path.join(OUT_PATH, "results_live_advtrain_perc")

DB_ANALYSIS_BASE = 'db_analysis_pypi_live_malwarebench_base.json'
DB_ANALYSIS_ADVTRAIN = 'db_analysis_pypi_live_malwarebench_advtrain.json'
DB_ANALYSIS_ADVTRAIN_PERC = 'db_analysis_pypi_live_malwarebench_advtrain_{perc}.json'

PACKAGE_INFO_PATH = 'db_new_packages_pypi_live_experiments.json'   # NOTE: Changed!

# Security advisories
OSSF_PACKAGES_INFO_PATH = os.path.join(OUT_PATH, "ossf-malicious-packages/osv/malicious/pypi/")
PYPI_MALWARE_REPORTS_PATH = os.path.join(OUT_PATH, "pypi-observation-reports-private/observations")

RESULTS_ANALYSIS = "results_analysis_pypi_live.json"

# DATASET_NEW_PACKAGES = os.path.join(OUT_PATH, 'dataset_pypi_live_clean.csv')
# DATASET_NEW_PACKAGES_SOA_FEATURES = os.path.join(OUT_PATH, 'dataset_pypi_live_soa_features.csv')
# LABELS_PACKAGES_PATH = '/home/mluser/workspace/adv_pkg/labels_packages_pypi_live_final.json'

# LIVE1
PACKAGE_INFO_LIVE1_PATH = os.path.join(OUT_PATH, 'db_new_packages_pypi_live1.json')
DATASET_NEW_PACKAGES_LIVE1 = os.path.join(OUT_PATH, 'dataset_pypi_live1.csv')
DATASET_NEW_PACKAGES_SOA_FEATURES_LIVE1 = os.path.join(OUT_PATH, 'dataset_pypi_live1_soa_features.csv')
LABELS_PACKAGES_LIVE1_PATH = os.path.join(OUT_PATH, 'labels_packages_pypi_live1.json')

# LIVE2
PACKAGE_INFO_LIVE2_PATH = os.path.join(OUT_PATH, 'db_new_packages_pypi_live2.json')
DATASET_NEW_PACKAGES_LIVE2 = os.path.join(OUT_PATH, 'dataset_pypi_live2.csv')
DATASET_NEW_PACKAGES_SOA_FEATURES_LIVE2 = os.path.join(OUT_PATH, 'dataset_pypi_live2_soa_features.csv')
LABELS_PACKAGES_LIVE2_PATH = os.path.join(OUT_PATH, 'labels_packages_pypi_live2.json')

# ENTERPRISE
ENTERPRISE_DATASET_BASE_PATH = os.path.join(DATA_BASE_PATH, 'dataset_packages_enterprise')
ENTERPRISE_PACKAGES_INFO = os.path.join(OUT_PATH, 'packages_info_enterprise.json')
DATASET_ENTERPRISE = os.path.join(OUT_PATH, 'dataset_enterprise.csv')
DATASET_SOA_FEATURES_ENTERPRISE = os.path.join(OUT_PATH, 'dataset_enterprise_soa_features.csv')
LABELS_PACKAGES_ENTERPRISE_PATH = os.path.join(OUT_PATH, 'labels_packages_enterprise.json')

SOURCERANK_DB_PATH = os.path.join(OUT_PATH, "sourcerank_live.json")
THRESHOLD_SOURCERANK = 8


def filter_packages_sourcerank(dataset_path, threshold=THRESHOLD_SOURCERANK):
    dataset = pd.read_csv(dataset_path)
    # consider only goodware packages
    dataset_goodware = dataset[dataset['label'] == 0]
    dataset_malware = dataset[dataset['label'] == 1]
    packages = dataset_goodware['Package Name'].values.tolist()

    with open(SOURCERANK_DB_PATH, 'r') as file:
        sourcerank_db = json.load(file)

    selected_packages = []
    for package in packages:
        if package in sourcerank_db and sourcerank_db[package] > threshold:
            selected_packages.append(package)

    # filter dataset
    dataset_goodware_filtered = dataset[dataset['Package Name'].isin(selected_packages)]
    dataset_filtered = pd.concat([dataset_goodware_filtered, dataset_malware], ignore_index=True)
    dataset_filtered.to_csv(dataset_path.replace('.csv', '_sourcerank.csv'), index=False)


def build_datasets_live(package_info_path, dataset_base_path, dataset_soa_features_path, labels_path):
    with open(package_info_path) as file:
        packages_info = json.load(file)
    
    to_remove = set()

    for pkg_name in packages_info:
        pkg_versions = packages_info[pkg_name].keys()
        pkg_version = sorted(pkg_versions)[-1]
        pkg_path = packages_info[pkg_name][pkg_version]
        pkg_name_version = f"{pkg_name}-{pkg_version}"

        # remove wheel packages
        # for file in os.listdir(pkg_path):
        #     if file == 'WHEEL':
        #         print(f"Found wheel package: {pkg_name_version}")
        #         to_remove.add(pkg_name)
        #         break

        # remove packages without metadata/installation files
        if not os.path.exists(os.path.join(pkg_path, 'setup.py')) and not os.path.exists(os.path.join(pkg_path, 'setup.cfg')) and \
                not os.path.exists(os.path.join(pkg_path, 'PKG-INFO')) and not os.path.exists(os.path.join(pkg_path, 'pyproject.toml')) and \
                not os.path.exists(os.path.join(pkg_path, 'setup.cfg')) and not os.path.exists(os.path.join(pkg_path, 'requirements.txt')):
            print(f"Found package without metadata/installation files: {pkg_name_version}")
            to_remove.add(pkg_name)
    
        # remove packages with no source code
        pkg = Package(pkg_name, pkg_version, pkg_type='pypi', path=pkg_path)
        if len(pkg.code_path) == 0 and len(pkg.metadata_path) == 0:
            print(f"Found package with no source code: {pkg_name_version}")
            to_remove.add(pkg_name)

    for pkg_name in to_remove:
        del packages_info[pkg_name]
    
    # update package info
    # with open(package_info_path, 'w') as file:
    #     json.dump(packages_info, file)
    
    packages = get_packages_from_json(packages_info, pkg_type='pypi')

    # build dataset
    feature_extractor = FeatureExtractor()
    df_vetting = feature_extractor.extract_features(packages, label=0, n_jobs=50)
    df_vetting.drop(columns=['label'], inplace=True)

    feature_extractor = FeatureExtractorSoaDetector()
    df_vetting_soa_features = feature_extractor.extract_features(packages, label=0, n_jobs=50)
    df_vetting_soa_features.drop(columns=['label'], inplace=True)

    common_packages = list(set(df_vetting['Package Name']).intersection(set(df_vetting_soa_features['Package Name'])))
    # print("Number of common packages in both datasets: ", len(common_packages))
    df_vetting = df_vetting[df_vetting['Package Name'].isin(common_packages)]
    df_vetting_soa_features = df_vetting_soa_features[df_vetting_soa_features['Package Name'].isin(common_packages)]

    df_vetting.to_csv(dataset_base_path, index=False)
    df_vetting_soa_features.to_csv(dataset_soa_features_path, index=False)
    
    get_ground_truth(dataset_base_path, labels_path)

    # add labels to the dataset
    with open(labels_path) as file:
        labels = json.load(file)
    
    df_vetting['label'] = df_vetting['Package Name'].map(labels)
    df_vetting_soa_features['label'] = df_vetting_soa_features['Package Name'].map(labels)

    df_vetting.to_csv(dataset_base_path, index=False)
    df_vetting_soa_features.to_csv(dataset_soa_features_path, index=False)


def build_datasets_live1():
    build_datasets_live(PACKAGE_INFO_LIVE1_PATH, DATASET_NEW_PACKAGES_LIVE1, DATASET_NEW_PACKAGES_SOA_FEATURES_LIVE1, LABELS_PACKAGES_LIVE1_PATH)


def build_datasets_live2():
    build_datasets_live(PACKAGE_INFO_LIVE2_PATH, DATASET_NEW_PACKAGES_LIVE2, DATASET_NEW_PACKAGES_SOA_FEATURES_LIVE2, LABELS_PACKAGES_LIVE2_PATH)


def build_datasets_enterprise():
    # create dataset info
    if not os.path.isfile(ENTERPRISE_PACKAGES_INFO):
        packages_info = dict()
        for pkg_name_version in os.listdir(ENTERPRISE_DATASET_BASE_PATH):
            pkg_name, pkg_version = pkg_name_version.rsplit('-', 1)
            pkg_path = os.path.join(ENTERPRISE_DATASET_BASE_PATH, pkg_name_version)
            if not os.path.isdir(pkg_path):
                continue
            if pkg_name not in packages_info:
                packages_info[pkg_name] = dict()
            packages_info[pkg_name][pkg_version] = pkg_path

        with open(ENTERPRISE_PACKAGES_INFO, 'w') as file:
            json.dump(packages_info, file)

    build_datasets_live(ENTERPRISE_PACKAGES_INFO, DATASET_ENTERPRISE, DATASET_SOA_FEATURES_ENTERPRISE, LABELS_PACKAGES_ENTERPRISE_PATH)


def get_threshold_at_fpr(labels, scores, max_fpr):
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores, drop_intermediate=True)
    best_threshold = thresholds[0]

    for fpr, threshold in zip(fpr_list[1:], thresholds[1:]):
        if fpr <= max_fpr:
            best_threshold = threshold

    return best_threshold


def get_packages_form_folder(base_path, pkg_type='pypi'):
    assert pkg_type in ['pypi', 'npm']
    packages = []

    for pkg in sorted([obj for obj in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, obj)) and not obj.startswith('.')]):
        pkg_name, pkg_version = pkg.rsplit('-', 1)
        packages.append(Package(pkg_name, pkg_version, pkg_type=pkg_type, path=os.path.join(base_path, pkg)))

    return packages


def get_packages_from_json(packages_info, pkg_type='pypi'):
    assert pkg_type in ['pypi', 'npm']
    packages = []

    for pkg_name in packages_info:
        for pkg_version, pkg_path in packages_info[pkg_name].items():
            packages.append(Package(pkg_name, pkg_version, pkg_type=pkg_type, path=pkg_path))

    # for pkg_name in packages_info:
    #     # for pkg_version, pkg_path in packages_info[pkg_name].items():
    #     pkg_versions = packages_info[pkg_name].keys()
    #     pkg_version = sorted(pkg_versions)[-1]  # take the latest version
    #     pkg_path = packages_info[pkg_name][pkg_version]
    #     packages.append(Package(pkg_name, pkg_version, pkg_type=pkg_type, path=pkg_path))

    return packages


def analyze_packages(model_type, db_analysis_path, use_soa_features=True, max_fpr=0.01, filter_sourcerank=True):
    if use_soa_features:
        dataset_new_packages_path = DATASET_NEW_PACKAGES_SOA_FEATURES_LIVE1
        dataset_path = DATASET_SOA_FEATURES_PATH
    else:
        dataset_new_packages_path = DATASET_NEW_PACKAGES_LIVE1
        dataset_path = DATASET_PATH
    
    if filter_sourcerank:
        dataset_new_packages_sourcerank_path = dataset_new_packages_path.replace('.csv', '_sourcerank.csv')
        if not os.path.isfile(dataset_new_packages_sourcerank_path):
            filter_packages_sourcerank(dataset_new_packages_path, threshold=THRESHOLD_SOURCERANK)
        dataset_new_packages_path = dataset_new_packages_sourcerank_path
    
    dataset_live = pd.read_csv(dataset_new_packages_path)
    
    test_labels = dataset_live['label'].values.tolist()
    if "label" == dataset_live.columns[-1]:
        dataset_live.drop(columns=['label'], inplace=True)
    # print(f"New packages dataset path: {dataset_new_packages_path}, shape: {dataset.shape}")
    package_names = dataset_live['Package Name']
    data = dataset_live.drop(columns=['Package Name']).to_numpy()

    if model_type == 'base':
        model_path = ML_MODEL_SOA_FEATURES_DEFAULT_PATH if use_soa_features else ML_MODEL_DEFAULT_PATH
    elif model_type == 'advtrain':
        model_path = ADVTRAIN_MODEL_SOA_FEATURES_DEFAULT_PATH if use_soa_features else ADVTRAIN_MODEL_DEFAULT_PATH
    elif model_type.startswith('advtrain_perc'):
        perc = int(model_type.split('_')[-1])
        if use_soa_features:
            model_path = ADVTRAIN_MODEL_SOA_FEATURES_PERC_PATH_SORTED.format(perc=perc)
        else:
            model_path = ADVTRAIN_MODEL_PERC_PATH_SORTED.format(perc=perc)

    # print(f"Model path: {model_path}")
    model = joblib.load(model_path)
    
    # load training set and compute scores
    training_set = pd.read_csv(dataset_path)
    # print(f"Train dataset path: {dataset_path}, shape: {training_set.shape}")
    training_labels = training_set['label'].values.tolist()
    training_data = training_set.drop(columns=['Package Name', 'label']).to_numpy()
    training_scores = model.predict_proba(training_data)[:, 1]

    test_scores = model.predict_proba(data)[:, 1]

    threshold = get_threshold_at_fpr(test_labels, test_scores, max_fpr)
    predictions = model.predict_proba(data)[:, 1] > threshold

    db_analysis = dict()
    for pkg, prediction in zip(package_names, predictions):
        pred_label = "malware" if prediction == 1 else "goodware"
        db_analysis[pkg] = pred_label
        # print(f"Package: {pkg} - Prediction: {pred_label}")
        # if prediction == 1:
        #     print(f"[WARN] Found new potential malware package: {pkg}")
        #     pass

    with open(db_analysis_path, 'w') as file:
        json.dump(db_analysis, file)


def analyze_packages_cmp(model_type, db_analysis_path, max_fpr=0.01, dataset_live_type='live1', use_final_models=False, filter_sourcerank=True):
    
    if model_type in ['base', 'advtrain']:
        if dataset_live_type == 'live1':
            dataset_new_packages_path = DATASET_NEW_PACKAGES_LIVE1
        elif dataset_live_type == 'live2':
            dataset_new_packages_path = DATASET_NEW_PACKAGES_LIVE2
        elif dataset_live_type == 'enterprise':
            dataset_new_packages_path = DATASET_ENTERPRISE

    elif model_type in ['soa_features', 'soa_features_advtrain']:
        if dataset_live_type == 'live1':
            dataset_new_packages_path = DATASET_NEW_PACKAGES_SOA_FEATURES_LIVE1
        elif dataset_live_type == 'live2':
            dataset_new_packages_path = DATASET_NEW_PACKAGES_SOA_FEATURES_LIVE2
        elif dataset_live_type == 'enterprise':
            dataset_new_packages_path = DATASET_SOA_FEATURES_ENTERPRISE

    if dataset_live_type == 'enterprise':
        dataset_path = DATASET_SOA_FEATURES_PATH
    else:
        dataset_path = dataset_new_packages_path

    if filter_sourcerank:
        dataset_new_packages_sourcerank_path = dataset_new_packages_path.replace('.csv', '_sourcerank.csv')
        if not os.path.isfile(dataset_new_packages_sourcerank_path):
            filter_packages_sourcerank(dataset_new_packages_path, threshold=THRESHOLD_SOURCERANK)
        dataset_new_packages_path = dataset_new_packages_sourcerank_path

    dataset = pd.read_csv(dataset_new_packages_path)
    test_labels = dataset['label'].values.tolist()
    if 'label' in dataset.columns:
        dataset.drop(columns=['label'], inplace=True)
    package_names = dataset['Package Name']
    data = dataset.drop(columns=['Package Name']).to_numpy()

    if model_type == 'base':
        model_path = ML_MODEL_DEFINITIVE_DEFAULT_PATH if use_final_models else ML_MODEL_DEFAULT_PATH
    elif model_type == 'advtrain':
        model_path = ADVTRAIN_MODEL_DEFINITIVE_DEFAULT_PATH if use_final_models else ADVTRAIN_MODEL_DEFAULT_PATH
    elif model_type == 'soa_features':
        model_path = ML_MODEL_SOA_FEATURES_DEFINITIVE_DEFAULT_PATH if use_final_models else ML_MODEL_SOA_FEATURES_DEFAULT_PATH
    elif model_type == 'soa_features_advtrain':
        model_path = ADVTRAIN_MODEL_SOA_FEATURES_DEFINITIVE_DEFAULT_PATH if use_final_models else ADVTRAIN_MODEL_SOA_FEATURES_DEFAULT_PATH

    model = joblib.load(model_path)
    
    # load training set, compute scores and use them to compute the threshold at 1% FPR
    training_set = pd.read_csv(dataset_path)
    training_labels = training_set['label'].values.tolist()
    training_data = training_set.drop(columns=['Package Name', 'label']).to_numpy()
    training_scores = model.predict_proba(training_data)[:, 1]

    test_scores = model.predict_proba(data)[:, 1]

    threshold = get_threshold_at_fpr(training_labels, training_scores, max_fpr)
    predictions = model.predict_proba(data)[:, 1] > threshold

    db_analysis = dict()
    for pkg, prediction in zip(package_names, predictions):
        pred_label = "malware" if prediction == 1 else "goodware"
        db_analysis[pkg] = pred_label
        # print(f"Package: {pkg} - Prediction: {pred_label}")
        # if prediction == 1:
        #     print(f"[WARN] Found new potential malware package: {pkg}")
        #     pass

    with open(db_analysis_path, 'w') as file:
        json.dump(db_analysis, file)


def analyze_package_advisories(package):
    pkg_name, pkg_version = package.rsplit('-', 1)
    pkg_name = pkg_name.lower()
    
    # check if the package has been reported by security advisories
    package_reported_ossf = set([pkg.lower() for pkg in sorted(os.listdir(OSSF_PACKAGES_INFO_PATH))])
    reported_ossf = pkg_name in package_reported_ossf

    package_reported_pypi = set([pkg.lower() for pkg in sorted(os.listdir(PYPI_MALWARE_REPORTS_PATH))])
    reported_pypi = pkg_name in package_reported_pypi

    return reported_ossf, reported_pypi


def process_model_results(db_analysis, package, label):
    if package not in db_analysis:
        return "none"
    pred_label = db_analysis[package]

    if pred_label == 'malware' and label == 0:
        result = "fp"
    elif pred_label == 'malware' and label == 1:
        result = "tp"
    elif pred_label == 'goodware' and label == 1:
        result = "fn"
    elif pred_label == 'goodware' and label == 0:
        result = "tn"

    return result


def analyze_results(target_perc=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], use_soa_features=True, max_fpr=0.01, advanced_analysis=False):
    with open(LABELS_PACKAGES_LIVE1_PATH) as file:
        packages_labels = json.load(file)
    
    packages, labels = list(packages_labels.keys()), list(packages_labels.values())
  
    if advanced_analysis:
        results = {
            "base": {
                "false_positives": [],
                "true_positives": [],
                "false_negatives": [],
                "true_negatives": []
            }
        }

        for perc in target_perc:
            results[f'advtrain_perc_{perc}'] = {
                "false_positives": [],
                "true_positives": [],
                "false_negatives": [],
                "true_negatives": []
            }
    else:    
        results = {
            "base": {
                "false_positives": 0,
                "true_positives": 0,
                "false_negatives": 0,
                "true_negatives": 0
            }
        }

        for perc in target_perc:
            results[f'advtrain_perc_{perc}'] = {
                "false_positives": 0,
                "true_positives": 0,
                "false_negatives": 0,
                "true_negatives": 0
            }

    for model_type in ['base'] + [f'advtrain_perc_{perc}' for perc in target_perc]:
        if model_type == 'base':
            db_analysis_path = os.path.join(RESULTS_BASE_PATH, DB_ANALYSIS_BASE)

            if use_soa_features:
                db_analysis_path = os.path.join(RESULTS_BASE_PATH, DB_ANALYSIS_BASE.replace('.json', '_soa_features.json'))
            db_analysis_path = db_analysis_path.replace('.json', f'_{max_fpr * 100}fpr.json')
        else:
            db_analysis_path = os.path.join(RESULTS_BASE_PATH, DB_ANALYSIS_ADVTRAIN_PERC.format(perc=model_type.split('_')[-1]))

            if use_soa_features:
                db_analysis_path = db_analysis_path.replace('.json', '_soa_features.json')
            db_analysis_path = db_analysis_path.replace('.json', f'_{max_fpr * 100}fpr.json')

        with open(db_analysis_path) as file:
            db_analysis = json.load(file)

        with multiprocessing.Pool(processes=60) as pool:
            results_packages = pool.starmap(process_model_results, [(db_analysis, pkg, label) for pkg, label in zip(packages, labels)])

        if advanced_analysis:
            for pkg, result in zip(packages, results_packages):
                if result == "fp":
                    results[model_type]['false_positives'].append(pkg)
                elif result == "tp":
                    results[model_type]['true_positives'].append(pkg)
                elif result == "fn":
                    results[model_type]['false_negatives'].append(pkg)
                elif result == "tn":
                    results[model_type]['true_negatives'].append(pkg)
        else:
            results[model_type]['false_positives'] += results_packages.count("fp")
            results[model_type]['true_positives'] += results_packages.count("tp")
            results[model_type]['false_negatives'] += results_packages.count("fn")
            results[model_type]['true_negatives'] += results_packages.count("tn")

    results_save_path = os.path.join(RESULTS_BASE_PATH, RESULTS_ANALYSIS)
    if use_soa_features:
        results_save_path = results_save_path.replace('.json', '_soa_features.json')
    if advanced_analysis:
        results_save_path = results_save_path.replace('.json', '_advanced.json')
    results_save_path = results_save_path.replace('.json', f'_{max_fpr * 100}fpr.json')
    with open(results_save_path, 'w') as file:
        json.dump(results, file)


def analyze_results_cmp_fpr(results_base_path, labels_path, max_fpr_list):
    with open(labels_path) as file:
        packages_labels = json.load(file)

    packages, labels = list(packages_labels.keys()), list(packages_labels.values())
    
    results = dict()
    for max_fpr in max_fpr_list:
        model_type = f'soa_features_advtrain_{max_fpr * 100}'
        results[model_type] = {
            "false_positives": 0,
            "true_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0
        }

        db_analysis_path = os.path.join(results_base_path, f'db_analysis_pypi_live_malwarebench_soa_features_advtrain_{max_fpr * 100}.json')

        with open(db_analysis_path) as file:
            db_analysis = json.load(file)

        with multiprocessing.Pool(processes=60) as pool:
            results_packages = pool.starmap(process_model_results, [(db_analysis, pkg, label) for pkg, label in zip(packages, labels)])        
        
        results[model_type]['false_positives'] += results_packages.count("fp")
        results[model_type]['true_positives'] += results_packages.count("tp")
        results[model_type]['false_negatives'] += results_packages.count("fn")
        results[model_type]['true_negatives'] += results_packages.count("tn")

        for pkg, result in zip(packages, results_packages):
            if result == "fp":
                results_advanced[model_type]['false_positives'].append(pkg)
            elif result == "tp":
                results_advanced[model_type]['true_positives'].append(pkg)
            elif result == "fn":
                results_advanced[model_type]['false_negatives'].append(pkg)

    results_save_path = os.path.join(results_base_path, "results_analysis.json")
    with open(results_save_path, 'w') as file:
        json.dump(results, file)


def analyze_results_all(results_base_path, labels_path, max_fpr=0.01):
    with open(labels_path) as file:
        packages_labels = json.load(file)

    packages, labels = list(packages_labels.keys()), list(packages_labels.values())
    
    results = {
        model_type: {
            "false_positives": 0,
            "true_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0
        }
        for model_type in ['base', 'soa_features', 'advtrain', 'soa_features_advtrain']
    }

    results_advanced = {
        model_type: {
            "false_positives": [],
            "true_positives": [],
            "false_negatives": []
        }
        for model_type in ['base', 'soa_features', 'advtrain', 'soa_features_advtrain']
    }

    db_analysis_basename = 'db_analysis_pypi_live_malwarebench_{model_type}.json'

    for model_type in ['base', 'soa_features', 'advtrain', 'soa_features_advtrain']:
        if model_type == 'base':
            db_analysis_path = os.path.join(results_base_path, db_analysis_basename.format(model_type='base'))
        if model_type == 'soa_features':
            db_analysis_path = os.path.join(results_base_path, db_analysis_basename.format(model_type='soa_features'))
        elif model_type == 'advtrain':
            db_analysis_path = os.path.join(results_base_path, db_analysis_basename.format(model_type='advtrain'))
        elif model_type == 'soa_features_advtrain':
            db_analysis_path = os.path.join(results_base_path, db_analysis_basename.format(model_type='soa_features_advtrain'))

        db_analysis_path = db_analysis_path.replace('.json', f'_{max_fpr * 100}fpr.json')

        with open(db_analysis_path) as file:
            db_analysis = json.load(file)

        with multiprocessing.Pool(processes=60) as pool:
            results_packages = pool.starmap(process_model_results, [(db_analysis, pkg, label) for pkg, label in zip(packages, labels)])        
        
        results[model_type]['false_positives'] += results_packages.count("fp")
        results[model_type]['true_positives'] += results_packages.count("tp")
        results[model_type]['false_negatives'] += results_packages.count("fn")
        results[model_type]['true_negatives'] += results_packages.count("tn")

        for pkg, result in zip(packages, results_packages):
            if result == "fp":
                results_advanced[model_type]['false_positives'].append(pkg)
            elif result == "tp":
                results_advanced[model_type]['true_positives'].append(pkg)
            elif result == "fn":
                results_advanced[model_type]['false_negatives'].append(pkg)

    results_save_path = os.path.join(results_base_path, "results_analysis.json")
    results_save_path = results_save_path.replace('.json', f'_{max_fpr * 100}fpr.json')
    with open(results_save_path, 'w') as file:
        json.dump(results, file)
    
    results_advanced_save_path = os.path.join(results_base_path, "results_analysis_advanced.json")
    results_advanced_save_path = results_advanced_save_path.replace('.json', f'_{max_fpr * 100}fpr.json')
    with open(results_advanced_save_path, 'w') as file:
        json.dump(results_advanced, file)


def analyze_results_advanced(results_base_path=RESULTS_BASE_PATH, max_fpr=0.01):
    results_advanced_save_path = os.path.join(results_base_path, "results_analysis_advanced.json")
    results_advanced_save_path = results_advanced_save_path.replace('.json', f'_{max_fpr * 100}fpr.json')
    with open(results_advanced_save_path, 'r') as file:
        results = json.load(file)

    tp_common_base_advtrain_soa_features = set(results['soa_features']['true_positives']).intersection(set(results['soa_features_advtrain']['true_positives']))
    tp_only_base_soa_features = set(results['soa_features']['true_positives']).difference(set(results['soa_features_advtrain']['true_positives']))
    tp_only_advtrain_soa_features = set(results['soa_features_advtrain']['true_positives']).difference(set(results['soa_features']['true_positives']))
    print(f"Number of TP in common between base and advtrain (SoA features): {len(tp_common_base_advtrain_soa_features)}")
    print(f"TP only in base (SoA features): {len(tp_only_base_soa_features)}")
    print(f"TP only in advtrain (SoA features): {len(tp_only_advtrain_soa_features)}")

    # analyze presence of obfuscation in the TP packages:
    # 1) Search all the TP packages in the dataset
    # 2) A package is considered obfuscated if it includes at least one obfuscated feature
    # 3) Compute the percentage of obfuscated packages in the TP packages
    dataset_path = DATASET_NEW_PACKAGES_SOA_FEATURES_LIVE1
    dataset_tp = pd.read_csv(dataset_path)
    obf_columns = [
        "count_basexx_source", "count_hex_source", "count_bytearray_source", "count_splitting_source", "count_xor_source", "count_api_obf_source",
        "count_basexx_metadata", "count_hex_metadata", "count_bytearray_metadata", "count_splitting_metadata", "count_xor_metadata", "count_api_obf_metadata"
    ]
    obfuscated_packages = dataset_tp[(dataset_tp[obf_columns] > 0).any(axis=1)]
    obfuscated_packages = obfuscated_packages['Package Name'].to_list()

    tp_common_base_advtrain_soa_features_obfuscated = [pkg for pkg in tp_common_base_advtrain_soa_features if pkg in obfuscated_packages]
    tp_only_base_soa_features_obfuscated = [pkg for pkg in tp_only_base_soa_features if pkg in obfuscated_packages]
    tp_only_advtrain_soa_features_obfuscated = [pkg for pkg in tp_only_advtrain_soa_features if pkg in obfuscated_packages]
    print(">> BASE vs ADVTRAIN (SOA FEATURES)")
    print(f"Number of TP in common: {len(tp_common_base_advtrain_soa_features_obfuscated)}")
    print(f"TP only in base {len(tp_only_base_soa_features_obfuscated)}:\n{tp_only_base_soa_features_obfuscated}")
    print(f"TP only in advtrain {len(tp_only_advtrain_soa_features_obfuscated)}:\n{tp_only_advtrain_soa_features_obfuscated}")

    tp_common_base_advtrain_soa_features_not_obfuscated = [pkg for pkg in tp_common_base_advtrain_soa_features if pkg not in obfuscated_packages]
    tp_only_base_soa_features_not_obfuscated = [pkg for pkg in tp_only_base_soa_features if pkg not in obfuscated_packages]
    tp_only_advtrain_soa_features_not_obfuscated = [pkg for pkg in tp_only_advtrain_soa_features if pkg not in obfuscated_packages]
    print(">> BASE vs ADVTRAIN (SOA FEATURES)")
    print(f"Number of TP in common: {len(tp_common_base_advtrain_soa_features_not_obfuscated)}")
    print(f"TP only in base {len(tp_only_base_soa_features_not_obfuscated)}:\n{tp_only_base_soa_features_not_obfuscated}")
    print(f"TP only in advtrain {len(tp_only_advtrain_soa_features_not_obfuscated)}:\n{tp_only_advtrain_soa_features_not_obfuscated}")

    assert len(tp_common_base_advtrain_soa_features) == len(tp_common_base_advtrain_soa_features_obfuscated) + len(tp_common_base_advtrain_soa_features_not_obfuscated)
    assert len(tp_only_base_soa_features) == len(tp_only_base_soa_features_obfuscated) + len(tp_only_base_soa_features_not_obfuscated)
    assert len(tp_only_advtrain_soa_features) == len(tp_only_advtrain_soa_features_obfuscated) + len(tp_only_advtrain_soa_features_not_obfuscated)


def compare_results(target_perc=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], use_soa_features=False, advaced_analysis=False, max_fpr=0.01):
    results_analysis_path = os.path.join(RESULTS_BASE_PATH, RESULTS_ANALYSIS)
    if use_soa_features:
        results_analysis_path = results_analysis_path.replace('.json', '_soa_features.json')
    if advaced_analysis:
        results_analysis_path = results_analysis_path.replace('.json', '_advanced.json')
    results_analysis_path = results_analysis_path.replace('.json', f'_{max_fpr * 100}fpr.json')
    with open(results_analysis_path) as file:
        results = json.load(file)

    # create and print a DataFrame with the results of all the models
    df_results = pd.DataFrame(columns=['Model', 'False Positives', 'True Positives', 'False Negatives', 'True Negatives', 'Precision', 'Recall', 'F1-Score'])
    for model_type in ['base'] + [f'advtrain_perc_{perc}' for perc in target_perc]:
        false_positives = results[model_type]['false_positives']
        if isinstance(false_positives, list):
            false_positives = len(false_positives)
        true_positives = results[model_type]['true_positives']
        if isinstance(true_positives, list):
            true_positives = len(true_positives)
        false_negatives = results[model_type]['false_negatives']
        if isinstance(false_negatives, list):
            false_negatives = len(false_negatives)
        true_negatives = results[model_type]['true_negatives']
        if isinstance(true_negatives, list):
            true_negatives = len(true_negatives)

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

        if len(df_results) == 0:
            df_results = pd.DataFrame({
                'Model': model_type,
                'False Positives': false_positives,
                'True Positives': true_positives,
                'False Negatives': false_negatives,
                'True Negatives': true_negatives,
                'Accuracy': accuracy * 100,
                'Precision': precision * 100,
                'Recall': recall * 100,
                'F1-Score': f1_score * 100
            }, index=[0])
        else:
            df_results = pd.concat([df_results, pd.DataFrame({
                'Model': model_type,
                'False Positives': false_positives,
                'True Positives': true_positives,
                'False Negatives': false_negatives,
                'True Negatives': true_negatives,
                'Accuracy': accuracy * 100,
                'Precision': precision * 100,
                'Recall': recall * 100,
                'F1-Score': f1_score * 100
            }, index=[0])], ignore_index=True)

        # df_results = df_results.append({
        #     'Model': model_type,
        #     'False Positives': len(false_positives),
        #     'True Positives': len(true_positives),
        #     'False Negatives': len(false_negatives),
        #     'True Negatives': len(true_negatives),
        #     'Precision': precision * 100,
        #     'Recall': recall * 100,
        #     'F1-Score': f1_score * 100
        # }, ignore_index=True)

    # print the results with two decimal places and do not print row index
    df_results = df_results.round(2)
    print(df_results.to_string(index=False))

    if advaced_analysis:
        assert isinstance(results['base']['true_positives'], list)
        assert isinstance(results['base']['false_positives'], list)
        assert isinstance(results['base']['false_negatives'], list)
        assert isinstance(results['base']['true_negatives'], list)

        true_positives_base = results['base']['true_positives']
        false_positives_base = results['base']['false_positives']

        fp_results = dict()

        for perc in target_perc:
            true_positives_advtrain_perc = results[f'advtrain_perc_{perc}']['true_positives']
            false_positives_advtrain_perc = results[f'advtrain_perc_{perc}']['false_positives']

            common_true_positives = set(true_positives_base).intersection(set(true_positives_advtrain_perc))
            print(f"Common true positives (perc={perc}) ({len(common_true_positives)}):")
            for pkg in common_true_positives:
                print(f"  - {pkg}")

            tp_only_base = set(true_positives_base).difference(set(true_positives_advtrain_perc))
            tp_only_advtrain = set(true_positives_advtrain_perc).difference(set(true_positives_base))
            
            print(f"True positives only in base model (perc={perc}) ({len(tp_only_base)}):")
            for pkg in tp_only_base:
                print(f"  - {pkg}")

            print(f"True positives only in advtrain model (perc={perc}) ({len(tp_only_advtrain)}):")
            for pkg in tp_only_advtrain:
                print(f"  - {pkg}")
            
            common_false_positives = set(false_positives_base).intersection(set(false_positives_advtrain_perc))
            print(f"Common false positives (perc={perc}) ({len(common_false_positives)}):")
            for pkg in common_false_positives:
                print(f"  - {pkg}")
            
            fp_only_base = set(false_positives_base).difference(set(false_positives_advtrain_perc))
            fp_only_advtrain = set(false_positives_advtrain_perc).difference(set(false_positives_base))

            print(f"False positives only in base model (perc={perc}) ({len(fp_only_base)}):")
            for pkg in fp_only_base:
                print(f"  - {pkg}")

            print(f"False positives only in advtrain model (perc={perc}) ({len(fp_only_advtrain)}):")
            for pkg in fp_only_advtrain:
                print(f"  - {pkg}")
        
            # save common FP and FP only in base/advtrain models in a single json file
            fp_results[f"base_advtrain_{perc}"] = {
                "common_fp": list(common_false_positives),
                "fp_only_base": list(fp_only_base),
                f"fp_only_advtrain_{perc}": list(fp_only_advtrain)
            }

        fp_results_path = f"fp_results.json"
        with open(fp_results_path, 'w') as file:
            json.dump(fp_results, file)


def compare_results_all(results_base_path, max_fpr=0.01):
    results_analysis_path = os.path.join(results_base_path, "results_analysis.json")
    results_analysis_path = results_analysis_path.replace('.json', f'_{max_fpr * 100}fpr.json')
    with open(results_analysis_path) as file:
        results = json.load(file)

    # create and print a DataFrame with the results of all the models
    df_results = pd.DataFrame(columns=['Model', 'False Positives', 'True Positives', 'False Negatives', 'True Negatives', 'Precision', 'Recall', 'F1-Score'])
    for model_type in ['base', 'soa_features', 'advtrain', 'soa_features_advtrain']:
        false_positives = results[model_type]['false_positives']
        if isinstance(false_positives, list):
            false_positives = len(false_positives)
        true_positives = results[model_type]['true_positives']
        if isinstance(true_positives, list):
            true_positives = len(true_positives)
        false_negatives = results[model_type]['false_negatives']
        if isinstance(false_negatives, list):
            false_negatives = len(false_negatives)
        true_negatives = results[model_type]['true_negatives']
        if isinstance(true_negatives, list):
            true_negatives = len(true_negatives)

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

        if len(df_results) == 0:
            df_results = pd.DataFrame({
                'Model': model_type,
                'False Positives': false_positives,
                'True Positives': true_positives,
                'False Negatives': false_negatives,
                'True Negatives': true_negatives,
                'Accuracy': accuracy * 100,
                'Precision': precision * 100,
                'Recall': recall * 100,
                'F1-Score': f1_score * 100
            }, index=[0])
        else:
            df_results = pd.concat([df_results, pd.DataFrame({
                'Model': model_type,
                'False Positives': false_positives,
                'True Positives': true_positives,
                'False Negatives': false_negatives,
                'True Negatives': true_negatives,
                'Accuracy': accuracy * 100,
                'Precision': precision * 100,
                'Recall': recall * 100,
                'F1-Score': f1_score * 100
            }, index=[0])], ignore_index=True)

    # results with two decimal places and do not print row index
    df_results = df_results.round(2)
    # print(df_results.to_string(index=False))
    with open(os.path.join(results_base_path, f"results_{max_fpr * 100}fpr.txt"), 'w') as file:
        file.write(df_results.to_string(index=False))


def compute_results_fpr(results_base_path, max_fpr_list=[0.001, 0.01, 0.1]):
    results_analysis_path = os.path.join(results_base_path, "results_analysis.json")
    with open(results_analysis_path) as file:
        results = json.load(file)

    # create and print a DataFrame with the results of all the models
    df_results = pd.DataFrame(columns=['Model', 'False Positives', 'True Positives', 'False Negatives', 'True Negatives', 'Precision', 'Recall', 'F1-Score'])
    for max_fpr in max_fpr_list:
        model_type = f'soa_features_advtrain_{max_fpr * 100}'
        false_positives = results[model_type]['false_positives']
        if isinstance(false_positives, list):
            false_positives = len(false_positives)
        true_positives = results[model_type]['true_positives']
        if isinstance(true_positives, list):
            true_positives = len(true_positives)
        false_negatives = results[model_type]['false_negatives']
        if isinstance(false_negatives, list):
            false_negatives = len(false_negatives)
        true_negatives = results[model_type]['true_negatives']
        if isinstance(true_negatives, list):
            true_negatives = len(true_negatives)

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

        if len(df_results) == 0:
            df_results = pd.DataFrame({
                'Model': {max_fpr * 100},
                'False Positives': false_positives,
                'True Positives': true_positives,
                'False Negatives': false_negatives,
                'True Negatives': true_negatives,
                'Accuracy': accuracy * 100,
                'Precision': precision * 100,
                'Recall': recall * 100,
                'F1-Score': f1_score * 100
            }, index=[0])
        else:
            df_results = pd.concat([df_results, pd.DataFrame({
                'Model': {max_fpr * 100},
                'False Positives': false_positives,
                'True Positives': true_positives,
                'False Negatives': false_negatives,
                'True Negatives': true_negatives,
                'Accuracy': accuracy * 100,
                'Precision': precision * 100,
                'Recall': recall * 100,
                'F1-Score': f1_score * 100
            }, index=[0])], ignore_index=True)
    
    # print the results with two decimal places and do not print row index
    df_results = df_results.round(2)
    print(df_results.to_string(index=False))


def get_ground_truth(dataset_path, labels_save_path):
    dataset = pd.read_csv(dataset_path)
    packages = dataset['Package Name'].values.tolist()

    with multiprocessing.Pool(processes=60) as pool:
        results = pool.map(analyze_package_advisories, packages)

    packages_labels = dict()
    """
    - reported by OSSF OR reported by PyPI: MALWARE (1)
    - else: GOODWARE (0)
    """
    for pkg, (reported_ossf, reported_pypi) in zip(packages, results):
        label = 1 if reported_ossf or reported_pypi else 0
        packages_labels[pkg] = label

    with open(labels_save_path, 'w') as file:
        json.dump(packages_labels, file)
    
    # add labels to the dataset
    dataset['label'] = dataset['Package Name'].apply(lambda x: packages_labels.get(x, 0))
    dataset.to_csv(dataset_path, index=False)
    
    return packages_labels


def experiments_advtrain_perc(soa_features=True, max_fpr=0.01, advanced_analysis=False):
    RESULTS_BASE_PATH = os.path.join(OUT_PATH, "results_live_cmp_advtrain")
    if soa_features:
        RESULTS_BASE_PATH += '_soa_features'

    if not os.path.exists(RESULTS_BASE_PATH):
        os.makedirs(RESULTS_BASE_PATH)

    db_analysis_base = os.path.join(RESULTS_BASE_PATH, DB_ANALYSIS_BASE)
    if soa_features:
        db_analysis_base = db_analysis_base.replace('.json', '_soa_features.json')
    db_analysis_base = db_analysis_base.replace('.json', f'_{max_fpr * 100}fpr.json')
    analyze_packages('base', db_analysis_path=db_analysis_base, use_soa_features=soa_features, max_fpr=max_fpr)
    for perc in TARGET_PERC:
        db_analysis_path = os.path.join(RESULTS_BASE_PATH, DB_ANALYSIS_ADVTRAIN_PERC.format(perc=perc))
        if soa_features:
            db_analysis_path = db_analysis_path.replace('.json', '_soa_features.json')

        db_analysis_path = db_analysis_path.replace('.json', f'_{max_fpr * 100}fpr.json')
        analyze_packages(f'advtrain_perc_{perc}', db_analysis_path=db_analysis_path, use_soa_features=soa_features, max_fpr=max_fpr)

    analyze_results(target_perc=TARGET_PERC, use_soa_features=soa_features, max_fpr=max_fpr, advanced_analysis=advanced_analysis)

    results_analysis_path = os.path.join(RESULTS_BASE_PATH, RESULTS_ANALYSIS)
    if soa_features:
        results_analysis_path = results_analysis_path.replace('.json', '_soa_features.json')
    results_analysis_path = results_analysis_path.replace('.json', f'_{max_fpr * 100}fpr.json')
    
    compare_results(target_perc=TARGET_PERC, use_soa_features=soa_features, advanced_analysis=advanced_analysis, max_fpr=max_fpr)


def experiments_real_world_live1(max_fpr=0.01):
    labels_path = LABELS_PACKAGES_LIVE1_PATH

    RESULTS_BASE_PATH = os.path.join(OUT_PATH, f"results_real_world_live1")

    if not os.path.exists(RESULTS_BASE_PATH):
        os.makedirs(RESULTS_BASE_PATH)
    
    db_analysis_basename = 'db_analysis_pypi_live_malwarebench_{model_type}.json'

    db_analysis_base_path = os.path.join(RESULTS_BASE_PATH, db_analysis_basename.format(model_type='base'))
    db_analysis_base_path = db_analysis_base_path.replace('.json', f'_{max_fpr * 100}fpr.json')
    analyze_packages_cmp('base', db_analysis_path=db_analysis_base_path, max_fpr=max_fpr, dataset_live_type='live1')

    db_analysis_soa_features_path = os.path.join(RESULTS_BASE_PATH, db_analysis_basename.format(model_type='soa_features'))
    db_analysis_soa_features_path = db_analysis_soa_features_path.replace('.json', f'_{max_fpr * 100}fpr.json')
    analyze_packages_cmp('soa_features', db_analysis_path=db_analysis_soa_features_path, max_fpr=max_fpr, dataset_live_type='live1')

    db_analysis_adv_train_path = os.path.join(RESULTS_BASE_PATH, db_analysis_basename.format(model_type='advtrain'))
    db_analysis_adv_train_path = db_analysis_adv_train_path.replace('.json', f'_{max_fpr * 100}fpr.json')
    analyze_packages_cmp('advtrain', db_analysis_path=db_analysis_adv_train_path, max_fpr=max_fpr, dataset_live_type='live1')

    db_analysis_soa_features_advtrain_path = os.path.join(RESULTS_BASE_PATH, db_analysis_basename.format(model_type='soa_features_advtrain'))
    db_analysis_soa_features_advtrain_path = db_analysis_soa_features_advtrain_path.replace('.json', f'_{max_fpr * 100}fpr.json')
    analyze_packages_cmp('soa_features_advtrain', db_analysis_path=db_analysis_soa_features_advtrain_path, max_fpr=max_fpr, dataset_live_type='live1')

    analyze_results_all(results_base_path=RESULTS_BASE_PATH, labels_path=labels_path, max_fpr=max_fpr)

    analyze_results_advanced(results_base_path=RESULTS_BASE_PATH, max_fpr=max_fpr)

    compare_results_all(results_base_path=RESULTS_BASE_PATH, max_fpr=max_fpr)


def experiments_real_world_live2(max_fpr=0.01):
    labels_path = LABELS_PACKAGES_LIVE2_PATH

    for final_models in [False, True]:
        RESULTS_BASE_PATH = os.path.join(OUT_PATH, f"results_real_world_live2")
        if final_models:
            RESULTS_BASE_PATH += '_final_models'

        if not os.path.exists(RESULTS_BASE_PATH):
            os.makedirs(RESULTS_BASE_PATH)
        
        db_analysis_basename = 'db_analysis_pypi_live_malwarebench_{model_type}.json'

        db_analysis_base_path = os.path.join(RESULTS_BASE_PATH, db_analysis_basename.format(model_type='base'))
        db_analysis_base_path = db_analysis_base_path.replace('.json', f'_{max_fpr * 100}fpr.json')
        analyze_packages_cmp('base', db_analysis_path=db_analysis_base_path, max_fpr=max_fpr, dataset_live_type='live2', use_final_models=final_models)

        db_analysis_soa_features_path = os.path.join(RESULTS_BASE_PATH, db_analysis_basename.format(model_type='soa_features'))
        db_analysis_soa_features_path = db_analysis_soa_features_path.replace('.json', f'_{max_fpr * 100}fpr.json')
        analyze_packages_cmp('soa_features', db_analysis_path=db_analysis_soa_features_path, max_fpr=max_fpr, dataset_live_type='live2', use_final_models=final_models)

        db_analysis_adv_train_path = os.path.join(RESULTS_BASE_PATH, db_analysis_basename.format(model_type='advtrain'))
        db_analysis_adv_train_path = db_analysis_adv_train_path.replace('.json', f'_{max_fpr * 100}fpr.json')
        analyze_packages_cmp('advtrain', db_analysis_path=db_analysis_adv_train_path, max_fpr=max_fpr, dataset_live_type='live2', use_final_models=final_models)

        db_analysis_soa_features_advtrain_path = os.path.join(RESULTS_BASE_PATH, db_analysis_basename.format(model_type='soa_features_advtrain'))
        db_analysis_soa_features_advtrain_path = db_analysis_soa_features_advtrain_path.replace('.json', f'_{max_fpr * 100}fpr.json')
        analyze_packages_cmp('soa_features_advtrain', db_analysis_path=db_analysis_soa_features_advtrain_path, max_fpr=max_fpr, dataset_live_type='live2', use_final_models=final_models)

        analyze_results_all(results_base_path=RESULTS_BASE_PATH, labels_path=labels_path, max_fpr=max_fpr)

        compare_results_all(results_base_path=RESULTS_BASE_PATH, max_fpr=max_fpr)


def experiments_case_study_fpr(dataset_type, max_fpr_list):
    assert dataset_type in ['live2', 'enterprise']
    if dataset_type == 'enterprise':
        if not os.path.isfile(DATASET_ENTERPRISE) and not os.path.isfile(DATASET_SOA_FEATURES_ENTERPRISE):
            build_datasets_enterprise()
        labels_path = LABELS_PACKAGES_ENTERPRISE_PATH

        RESULTS_BASE_PATH = os.path.join(OUT_PATH, "results_use_case_enterprise")
    else:
        if not os.path.isfile(DATASET_NEW_PACKAGES_LIVE2) and not os.path.isfile(DATASET_NEW_PACKAGES_SOA_FEATURES_LIVE2):
            build_datasets_live2()
        labels_path = LABELS_PACKAGES_LIVE2_PATH

        RESULTS_BASE_PATH = os.path.join(OUT_PATH, "results_use_case_pypi")

    if not os.path.exists(RESULTS_BASE_PATH):
        os.makedirs(RESULTS_BASE_PATH)
    
    db_analysis_basename = 'db_analysis_pypi_live_malwarebench_soa_features_advtrain_{max_fpr}.json'

    for fpr in max_fpr_list:
        db_analysis_soa_features_advtrain_path = db_analysis_soa_features_advtrain_path.format(max_fpr=f'_{fpr * 100}fpr.json')
        analyze_packages_cmp('soa_features_advtrain', db_analysis_path=db_analysis_soa_features_advtrain_path, max_fpr=max_fpr, dataset_live_type="live2", use_final_models=True, filter_sourcerank=True)

    analyze_results_cmp_fpr(results_base_path=RESULTS_BASE_PATH, labels_path=labels_path, max_fpr_list=max_fpr_list)

    compute_results_fpr(results_base_path=RESULTS_BASE_PATH, max_fpr_list=max_fpr_list)


def experiments_pypi_case_study():
    experiments_use_case_fpr(dataset_type='live2', max_fpr_list=[0.0005, 0.001, 0.01, 0.1, 0.3])


def experiments_enterprise_case_study():
    experiments_case_study_fpr(dataset_type='enterprise', max_fpr_list=[0.001, 0.01, 0.1, 0.3])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze new packages')
    parser.add_argument('experiment', type=str, help='Type of experiment to run. Options: advtrain-perc, real-world-live1, real-world-live2, pypi-use-case, enterprise-use-case')
    parser.add_argument('--max-fpr', default=0.01, help='Max FPR to use for tuning the detectors. Default is 0.01 (1%).')
    args = parser.parse_args()
    
    if not os.path.isfile(DATASET_NEW_PACKAGES_SOA_FEATURES_LIVE1) or not os.path.isfile(DATASET_NEW_PACKAGES_LIVE1):
        build_datasets_live1()

    if not os.path.isfile(DATASET_NEW_PACKAGES_SOA_FEATURES_LIVE2) or not os.path.isfile(DATASET_NEW_PACKAGES_LIVE2):
        build_datasets_live2()
    
    if args.experiment == 'advtrain-perc':
        experiments_advtrain_perc(max_fpr=float(args.max_fpr), advanced_analysis=True)
    elif args.experiment == 'real-world-live1':
        experiments_real_world_live1(max_fpr=float(args.max_fpr))
    elif args.experiment == 'real-world-live2':
        experiments_real_world_live2(max_fpr=float(args.max_fpr))
    elif args.experiment == 'pypi-case-study':
        experiments_pypi_case_study()
    elif args.experiment == 'enterprise-case-study':
        experiments_enterprise_case_study()
