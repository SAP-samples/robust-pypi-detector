from model import SklearnModel, GuarddogWrapper
from optimizer import Optimizer, OptimizerAdvanced
import multiprocessing
import os
import json
import argparse
import shutil
from utils import Package
import multiprocessing as mp

OUT_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/out'
DATA_BASE_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/data'
if not os.path.isdir(OUT_PATH):
    os.makedirs(OUT_PATH)
if not os.path.isdir(DATA_BASE_PATH):
    os.makedirs(DATA_BASE_PATH)

DATA_PATH = os.path.join(DATA_BASE_PATH, 'dataset_malware')
ADV_PACKAGES_PATH = os.path.join(DATA_BASE_PATH, 'data/dataset_adv')
OUT_RESULTS_FILEPATH = os.path.join(OUT_PATH, 'results.json')
RESULTS_DIR = os.path.join(OUT_PATH, 'eval_results')
NUM_ROUNDS_DEFAULT = 10
PACKAGES_TEMP_POOL = os.path.join(DATA_BASE_PATH, 'packages_pool')
NUM_FOLDS = 5


def run_attack(
    model_path,
    model_type='sklearn',
    data_path=DATA_PATH,
    adv_packages_path=ADV_PACKAGES_PATH,
    num_rounds=NUM_ROUNDS_DEFAULT,
    out_results_filepath=OUT_RESULTS_FILEPATH,
    log_base_path=None,
    pool_path=PACKAGES_TEMP_POOL,
    only_sr=False,
    overwrite=False,
    use_soa_features=False,
    debug=True
):
    if model_type == "sklearn":
        model = SklearnModel(model_path, use_soa_features=use_soa_features)
    elif model_type == "guarddog":
        model = GuarddogWrapper()
    else:
        raise Exception("Unsupported model type: {}".format(model_type))

    assert os.path.isdir(data_path)
    if not os.path.isdir(adv_packages_path):
        os.makedirs(adv_packages_path)

    malware_samples = sorted(os.listdir(data_path))

    packages = list()
    log_files = list()
    for sample in malware_samples:
        adv_pkg_path = os.path.join(adv_packages_path, sample)
        if not overwrite and os.path.isdir(adv_pkg_path):
            if debug:
                print(f"Skipping packages {sample}, already analyzed\n")
            continue
        shutil.copytree(os.path.join(data_path, sample), adv_pkg_path)
        pkg_name, pkg_version = sample.rsplit('-', 1)
        pkg = Package(pkg_name, pkg_version, pkg_type='pypi', path=adv_pkg_path)
        packages.append(pkg)
        log_files.append(os.path.join(log_base_path, f"{pkg_name}-{pkg_version}.log") if log_base_path else None)

    optimizer = Optimizer(model, num_rounds, pool_dir=pool_path, only_sr=only_sr, debug=True)
    # optimizer = OptimizerAdvanced(model, num_rounds=10, num_candidates=10, num_transformations_round=40, debug=True)

    # n_jobs = len(os.sched_getaffinity(0)) - 1
    # n_jobs = int(len(os.sched_getaffinity(0)) // NUM_FOLDS)
    n_jobs = 16
    with multiprocessing.Pool(processes=n_jobs) as pool:
        # results = pool.map(optimizer.optimize, packages)
        results = pool.starmap(optimizer.optimize, zip(packages, log_files))

    # results = []
    # for pkg, log_file in zip(packages, log_files):
    #     # print(f"Manipulating package: {pkg.name}-{pkg.version}")
    #     result_attack = optimizer.optimize(pkg, log_file)
    #     results.append(result_attack)

    with open(out_results_filepath, 'w') as out_file:
        for result in results:
            best_score, adv_pkg, num_queries, run_time, scores_trace = result
            info = {'sample': f'{adv_pkg.name}-{adv_pkg.version}', 'best_score': float(best_score),
                    'num_queries': num_queries, 'run_time': run_time, 'scores_trace': str(scores_trace)}
            out_file.write(json.dumps(info) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run adversarial attacks against a ML-based malicious package detector.')
    parser.add_argument('model_path', type=str, help='Path of the ML model')
    parser.add_argument('model_type', type=str, default='sklearn', help='Type of the ML model (sklearn or keras)')
    parser.add_argument('data_path', type=str, default=DATA_PATH, help='Folder containing the malware samples (packages) to be manipulated.')
    parser.add_argument('output_path', type=str, default=ADV_PACKAGES_PATH, help='Path where to save the output results (adversarial packages)')    
    parser.add_argument('num_rounds', type=int, default=NUM_ROUNDS_DEFAULT, help='Number of mutational rounds for the multi-round transformations.')
    parser.add_argument('results_path', type=str, default=OUT_RESULTS_FILEPATH, help='Path of the JSON file to store the results of adversarial attacks.')
    parser.add_argument('log_base_path', type=str, default=RESULTS_DIR, help='Base path to store the logs of the optimization.')
    parser.add_argument('pool_path', type=str, default=PACKAGES_TEMP_POOL, help='Temporary path to store the mutated packages.')
    parser.add_argument('--only-sr', action='store_true', help='Use only single-round transformations.')
    parser.add_argument('--soa-features', action='store_true', help='Whether to use the SoA features or the default ones.')
    args = parser.parse_args()

    # Workaround to avoid deadlock when using xgboost: https://github.com/dmlc/xgboost/issues/6617#issuecomment-781455825
    mp.set_start_method('spawn')

    run_attack(args.model_path, args.model_type, data_path=args.data_path, adv_packages_path=args.output_path, num_rounds=args.num_rounds,
               out_results_filepath=args.results_path, log_base_path=args.log_base_path, pool_path=args.pool_path, only_sr=args.only_sr,
               overwrite=True, use_soa_features=args.soa_features)
