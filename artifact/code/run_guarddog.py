from guarddog.scanners import PypiPackageScanner, NPMPackageScanner
import os
import argparse
import json
import multiprocessing
import pandas as pd


OUT_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/out'
DATA_BASE_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/data'

GOODWARE_PATH = os.path.join(DATA_BASE_PATH, 'dataset_goodware_pypi_malwarebench')
MALWARE_PATH = os.path.join(DATA_BASE_PATH, 'dataset_malware_pypi_malwarebench')
OUTPUT_PATH = os.path.join(OUT_PATH, 'scores_guarddog_malwarebench.json')
NUM_PROC = 32


GUARDDOG_RULES = [
    'clipboard-access',
    'cmd-overwrite',
    'code-execution',
    'dll-hijacking',
    'download-executable',
    'exec-base64',
    'exfiltrate-sensitive-data',
    'obfuscation',
    'shady-links',
    'silent-process-execution',
    'steganography'
]


def scan_package(package, package_path, ecosystem='pypi'):
    if ecosystem == 'pypi':
        scanner = PypiPackageScanner()
    elif ecosystem == 'npm':
        scanner = NpmPackageScanner()
    else:
        raise ValueError(f'Ecosystem not supported: {ecosystem}')

    return {'package': package} | scanner.scan_local(package_path)


def scan_packages_guarddog(base_path, ecosystem='pypi'):
    """
    Returns the feature representation of the input.

    Arguments:
        base_path: directory containing the packages to scan

    Returns:
        results: list of dictionaries containing the scan results
    """

    with multiprocessing.Pool(NUM_PROC) as pool:
        results = pool.starmap(scan_package,
                              [(package, os.path.join(base_path, package), 'pypi') for package in os.listdir(base_path) if not package.startswith('.') and os.path.isdir(os.path.join(base_path, package))])

    return results


def get_scores(results):
    scores = {}
    for result in results:
        scores[result['package']] = result['issues']

    return scores


def run():
    results_goodware = scan_packages_guarddog(GOODWARE_PATH, 'pypi')
    scores_goodware = get_scores(results_goodware)
    results_malware = scan_packages_guarddog(MALWARE_PATH, 'pypi')
    scores_malware = get_scores(results_malware)

    scores = scores_goodware | scores_malware
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(scores, f, indent=4)


if __name__ == '__main__':
    run()