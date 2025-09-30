import json
import os
import pandas as pd
import shlex
import shutil
import subprocess
import argparse


OUT_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/out'
DATA_BASE_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/data'

MODELS_BASE_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi{soa_feat}')
ML_MODELS = [
    ('guarddog', 'guarddog'),
    ('rf', 'sklearn'),
    ('dt', 'sklearn'),
    ('xgboost', 'sklearn')
]

MODELS_RETRAINED_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain{soa_feat}_{perc}_sorted')
MODELS_RETRAINED_FULL_PERC_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain{soa_feat}_full_{perc}_sorted')
MODELS_RETRAINED_FULL_BASE_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_full{soa_feat}')
ML_MODELS_RETRAINED = [
    ('xgboost', 'sklearn')
]

DATASET_PATH = os.path.join(OUT_PATH, 'dataset_pypi_malwarebench{soa_feat}.csv')
DATA_MAIN_PATH = os.path.join(DATA_BASE_PATH, 'dataset_malware_pypi_malwarebench')
DATA_FOLDS_TEST_BASE_PATH = os.path.join(DATA_BASE_PATH, 'dataset_test_folds_malwarebench{soa_feat}')
DATA_FOLDS_ADVTRAIN_BASE_PATH = os.path.join(DATA_BASE_PATH, 'dataset_advtrain_folds_malwarebench{soa_feat}')
DATA_ADV_BASE_PATH = os.path.join(DATA_BASE_PATH, 'dataset_adv_{model}_malwarebench{soa_feat}')
RESULTS_BASE_PATH = os.path.join(OUT_PATH, 'eval_results_malwarebench')
PACKAGES_TEMP_POOL = os.path.join(DATA_BASE_PATH, 'packages_pool_{model}_malwarebench{soa_feat}')


ADVTRAIN_SAMPLES_RATIO = 1.0  # ratio of malicious packages in the training set that are used for adv training

TARGET_PERC_RETRAINED = 20

DATASET_LIVE_PATH = os.path.join(OUT_PATH, 'dataset_pypi_live1.csv')
DATASET_INFO_LIVE_PATH = os.path.join(OUT_PATH, 'db_new_packages_pypi_live1.json')
DATA_TEST_BASE_PATH = os.path.join(DATA_BASE_PATH, 'dataset_test_malwarebench_adv')

NUM_FOLDS = 5
NUM_ROUNDS = 5


def build_test_sets(soa_features=False):
    dataset_path = DATASET_PATH.format(soa_feat='_soa_features' if soa_features else '')
    dataset = pd.read_csv(dataset_path)

    models_path = MODELS_BASE_PATH.format(soa_feat='_soa_features' if soa_features else '')

    for idx in range(NUM_FOLDS):
        with open(os.path.join(models_path, f"test_indices_fold{idx+1}.json")) as file:
            test_indices = json.load(file)
        test_samples = dataset[dataset.index.isin(test_indices) & dataset.label == 1]['Package Name'].to_list()

        data_folds_test_path = DATA_FOLDS_TEST_BASE_PATH.format(soa_feat='_soa_features' if soa_features else '')
        fold_path = os.path.join(data_folds_test_path, f"fold{idx+1}")
        if not os.path.isdir(fold_path):
            os.makedirs(fold_path)
            for sample in test_samples:
                shutil.copytree(os.path.join(DATA_MAIN_PATH, sample), os.path.join(fold_path, sample))


def build_adv_train_sets(soa_features=False):
    dataset_path = DATASET_PATH.format(soa_feat='_soa_features' if soa_features else '')
    dataset = pd.read_csv(dataset_path)

    for idx in range(NUM_FOLDS):
        models_path = MODELS_BASE_PATH.format(soa_feat='_soa_features' if soa_features else '')
        with open(os.path.join(models_path, f"test_indices_fold{idx+1}.json")) as file:
            test_indices = json.load(file)
        train_samples_malware = dataset[~dataset.index.isin(test_indices) & dataset.label == 1]['Package Name']

        num_samples_advtrain = min(int(len(train_samples_malware) * ADVTRAIN_SAMPLES_RATIO), len(train_samples_malware))
        train_samples_malware = train_samples_malware.sample(n=num_samples_advtrain, ignore_index=True, random_state=0).to_list()
        assert len(train_samples_malware) == num_samples_advtrain and len(train_samples_malware) == len(set(train_samples_malware))

        data_folds_advtrain_path = DATA_FOLDS_ADVTRAIN_BASE_PATH.format(soa_feat='_soa_features' if soa_features else '')
        fold_path = os.path.join(data_folds_advtrain_path, f"fold{idx+1}")
        if not os.path.isdir(fold_path):
            os.makedirs(fold_path)
            for sample in train_samples_malware:
                shutil.copytree(os.path.join(DATA_MAIN_PATH, sample), os.path.join(fold_path, sample))


def run_experiments(mode='test', soa_features=False):
    assert mode in ['test', 'adv_train', 'test_retrained']
    cmd_base = "python run_attack.py {only_sr}{model} {model_type} {data_path} {adv_data_path} {num_rounds} {results_path} {log_filepath} {pool_path}{soa_feat}"

    # if os.path.isdir(RESULTS_BASE_PATH):
    #     shutil.rmtree(RESULTS_BASE_PATH)
    #     os.makedirs(RESULTS_BASE_PATH)

    if mode == 'test':
        data_base_path = DATA_FOLDS_TEST_BASE_PATH.format(soa_feat='_soa_features' if soa_features else '')
        models_evaluated = ML_MODELS
        models_path = MODELS_BASE_PATH.format(soa_feat='_soa_features' if soa_features else '')
        build_test_sets(soa_features)
    elif mode == 'adv_train':
        data_base_path = DATA_FOLDS_ADVTRAIN_BASE_PATH.format(soa_feat='_soa_features' if soa_features else '')
        models_evaluated = ML_MODELS_RETRAINED
        models_path = MODELS_BASE_PATH.format(soa_feat='_soa_features' if soa_features else '')
        build_adv_train_sets(soa_features)
    else:  # test_retrained
        data_base_path = DATA_FOLDS_TEST_BASE_PATH.format(soa_feat='_soa_features' if soa_features else '')
        models_evaluated = ML_MODELS_RETRAINED
        models_path = MODELS_RETRAINED_PATH.format(soa_feat='_soa_features' if soa_features else '', perc=TARGET_PERC_RETRAINED)

    for model, model_type in models_evaluated:
        if mode == 'adv_train':
            model_name = model + '_advtrain'
        elif mode == 'test_retrained':
            model_name = model + f'_retrained_{TARGET_PERC_RETRAINED}'
            # NOTE: Changed the retrained model with the one re-trained on 50% of the adversarial examples
            # model_name = model + '_retrained_50_sorted'
        else:
            model_name = model
        data_adv_model = DATA_ADV_BASE_PATH.format(model=model_name, soa_feat='_soa_features' if soa_features else '')
        if os.path.isdir(data_adv_model):
            shutil.rmtree(data_adv_model)
        pool_path_base = PACKAGES_TEMP_POOL.format(model=model_name, soa_feat='_soa_features' if soa_features else '')
        if os.path.isdir(pool_path_base):
            shutil.rmtree(pool_path_base)
        os.makedirs(pool_path_base)
        for fold in range(NUM_FOLDS):
            model_path = os.path.join(models_path, f"result_{fold+1}", f"{model}_fold{fold+1}.joblib")
            data_path = os.path.join(data_base_path, f"fold{fold+1}")
            out_adv_path = os.path.join(data_adv_model, f"fold{fold+1}")
            results_path = os.path.join(RESULTS_BASE_PATH, f"result_{model_name}_fold{fold+1}{'_soa_features' if soa_features else ''}.json")
            log_filepath = os.path.join(RESULTS_BASE_PATH, f"log_{model_name}_fold{fold+1}{'_soa_features' if soa_features else ''}")

            if mode == 'adv_train':
                model_pool_path = os.path.join(pool_path_base, f"fold{fold+1}")
                os.makedirs(model_pool_path, exist_ok=True)
            else:
                model_pool_path = pool_path_base
            if not os.path.isdir(log_filepath):
                os.makedirs(log_filepath)
            only_sr = '--only-sr ' if model_type == 'guarddog' else ''

            cmd = cmd_base.format(model=model_path, model_type=model_type, data_path=data_path, adv_data_path=out_adv_path, num_rounds=NUM_ROUNDS,
                                  results_path=results_path, log_filepath=log_filepath, pool_path=model_pool_path, only_sr=only_sr,
                                  soa_feat=' --soa-features' if soa_features else '')
            subprocess.Popen(shlex.split(cmd))
            # subprocess.run(shlex.split(cmd))
            # run_attack(model_path, model_type='sklearn', data_path=data_path, adv_packages_path=out_adv_path,
            #            num_rounds=NUM_ROUNDS, out_results_filepath=results_path, overwrite=False, debug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('type', type=str, default="test", help='Type of experiment to run (test, adv_train, test_retrained)')
    parser.add_argument('--soa-features', action='store_true', help='Whether to use the SoA features')
    args = parser.parse_args()

    assert args.type in ['test', 'adv-train', 'test-retrained', 'comparison']
    if args.type == 'test':
        run_experiments('test', soa_features=args.soa_features)
    if args.type == 'adv-train':
        run_experiments('adv_train', soa_features=args.soa_features)
    if args.type == 'test-retrained':
        run_experiments('test_retrained', soa_features=args.soa_features)
