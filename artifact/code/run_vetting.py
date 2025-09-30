import argparse
from feature_extractor_soa_detector import FeatureExtractorSoaDetector
import joblib
import json
import os
from utils import Package
import pandas as pd
import requests
import re
import multiprocessing
import feedparser
import time
from datetime import datetime
import tarfile
import zipfile
import shutil
import hashlib
from packaging.version import Version

OUT_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/out'
DATA_BASE_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/data'
# temporary path to save the models to be released for the artifact
MODELS_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/models'

# ADVTRAIN_MODEL_SOA_PATH = os.path.join(OUT_PATH, 'training_results_malwarebench/training_results_pypi_advtrain_soa_features_definitive_full_20_sorted/xgboost_final_20.joblib')
ADVTRAIN_MODEL_SOA_PATH = os.path.join(MODELS_PATH, 'xgboost_final_20.joblib')

# PyPI feeds
PYPI_NEWEST_PACKAGES_FEED_URL = "https://pypi.org/rss/packages.xml"
PYPI_LATEST_UPDATED_FEED_URL = "https://pypi.org/rss/updates.xml"
WAIT_TIME = 60


def scan_new_packages(packages_db_path, packges_save_path, dataset_path, malware_found_path):

    if os.path.isfile(packages_db_path):
        with open(packages_db_path, 'r') as f:
            packages_db = json.load(f)
    else:
        packages_db = dict()

    if not os.path.isdir(packges_save_path):
        os.makedirs(packges_save_path)

    try:
        rss_new_published = True
        while True:
            url = PYPI_NEWEST_PACKAGES_FEED_URL if rss_new_published else PYPI_LATEST_UPDATED_FEED_URL

            try:
                packages_info = feedparser.parse(url)
            except Exception as e:
                print(f'Error: {e}')
                time.sleep(WAIT_TIME)
                continue

            if packages_info is None:
                print('Error: Failed to get new packages information from PyPI.')
                time.sleep(WAIT_TIME)
                continue

            pkg_name_path = set()
            for package_metadata in packages_info['entries']:
                pkg_date = str(datetime.strptime(package_metadata['published'], "%a, %d %b %Y %H:%M:%S GMT").date())
                pkg_name = package_metadata['title'].split(" ")[0]
                pkg_name_path.add((pkg_name, os.path.join(packges_save_path, pkg_date)))

            with multiprocessing.Pool(processes=20) as pool:
                res = pool.starmap(download_package, list(pkg_name_path))

            if os.path.isfile(dataset_path):
                df_packages = pd.read_csv(dataset_path)
                packages_in_dataset = set(df_packages['Package Name'].values.tolist())
            else:
                packages_in_dataset = set()
                df_packages = pd.DataFrame()
            
            packages_to_analyze = []                
            for pkg_path in res:
                if pkg_path:
                    pkg_name_version = pkg_path.rsplit('/', 1)[-1]
                    pkg_name, pkg_version = pkg_name_version.rsplit('-', 1)

                    if pkg_name_version not in packages_in_dataset:
                        packages_to_analyze.append(Package(pkg_name, pkg_version, pkg_type='pypi', path=pkg_path))

                    if pkg_name in packages_db:
                        packages_db[pkg_name][pkg_version] = pkg_path
                    else:
                        packages_db[pkg_name] = dict()
                        packages_db[pkg_name][pkg_version] = pkg_path

            with open(packages_db_path, 'w') as f:
                json.dump(packages_db, f)

            # packages_to_analyze contains the packages that need to be analyzed: packages not in the dataset and not analyzed yet
            if len(packages_to_analyze) > 0:
                feature_extractor = FeatureExtractorSoaDetector()
                df_new_packages = feature_extractor.extract_features(packages_to_analyze, label=0)
                df_new_packages.drop(columns=['label'], inplace=True)
                package_names = df_new_packages['Package Name']
                data = df_new_packages.drop(columns=['Package Name']).to_numpy()

                adv_train_model = joblib.load(ADVTRAIN_MODEL_SOA_PATH)
                advtrain_predictions = adv_train_model.predict(data).astype(int)

                # add label to the dataframe
                df_new_packages['label'] = advtrain_predictions.tolist()

                for pkg, prediction in zip(package_names, advtrain_predictions):
                    if prediction == 1:
                        print(f"[WARN] New malware: {pkg}")
                        print(f"{pkg}", file=open(malware_found_path, 'a+'))

                if len(df_new_packages) > 0:
                    if len(df_packages) > 0:
                        df_packages = pd.concat([df_packages, df_new_packages], ignore_index=True)
                    else:
                        df_packages = df_new_packages
                    df_packages.sort_values(by='Package Name', inplace=True)
                    df_packages.to_csv(dataset_path, index=False)

            time.sleep(WAIT_TIME)
            # alternate between the two RSS feeds
            rss_new_published = not rss_new_published
    except KeyboardInterrupt:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Stopped package collection on {now}.")
    # except Exception as e:
    #     print(f"Error when collecting the packages: {e}")


def download_package(package_name, out_base_path):
    pkg_info_url = f"https://pypi.org/pypi/{package_name}/json"
    pkg_info = requests.get(pkg_info_url).json()
    pkg_date = out_base_path.rsplit('/', 1)[-1]

    os.makedirs(out_base_path, exist_ok=True)

    download_url = None
    if 'releases' not in pkg_info:
        print(f"[WARN] No releases found for {package_name}")
        return

    if not pkg_info['releases']:
        return

    versions = list(pkg_info['releases'].keys())
    try:
        version.sort(key=Version)
    except:
        versions.sort()
    latest_version = versions[-1]

    if os.path.isdir(os.path.join(out_base_path, f"{package_name}-{latest_version}")):
        # print(f"[INFO] {package_name}-{latest_version} already downloaded")
        return

    print(f'New package {package_name}-{latest_version} published on {pkg_date}.')
    for pkg_file in pkg_info['releases'][latest_version]:
        # if pkg_file['filename'].endswith('.tar.gz') or pkg_file['filename'].endswith('.zip'):
        if pkg_file['python_version'] == 'source':
            download_url = pkg_file['url']
            package_ext = 'tar.gz' if pkg_file['filename'].endswith('.tar.gz') else 'zip'
            file_name = pkg_file['filename']

    # if download_url is None:
    #     print(f"[INFO] No sources found for {package_name}")
    #     if len(pkg_info['releases'][latest_version]) > 0 and pkg_info['releases'][latest_version][0]['packagetype'] == 'bdist_wheel':
    #         print(f"[INFO] Found wheel for {package_name}")
    #         download_url = pkg_info['releases'][latest_version][0]['url']
    #         package_ext = 'whl'
    #         file_name = pkg_info['releases'][latest_version][0]['filename']
    
    if download_url is None:
        print(f"[WARN] No source or wheel found for {package_name}")
        return

    # Send a GET request to the download URL
    response = requests.get(download_url)
    # Save the response content to a file
    pkg_path = os.path.join(out_base_path, file_name)
    try:
        with open(pkg_path, "wb") as file:
            file.write(response.content)
    except Exception:
        print(f"Failed to save {file_name}")
        return
    else:
        try:
            # create temporary directory to extract the package
            tmp_dir = os.path.join(os.getcwd(), f'tmp_{package_name}-{latest_version}')
            os.makedirs(tmp_dir, exist_ok=True)

            if package_ext == 'tar.gz':
                with tarfile.open(pkg_path, 'r:gz') as tar:
                    tar.extractall(tmp_dir)
            else:
                with zipfile.ZipFile(pkg_path, 'r') as zip_file:
                    zip_file.extractall(tmp_dir)

            extracted_dirs = [obj for obj in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, obj))]
            if len(extracted_dirs) != 1:
                # the source files are saved in tmp_dir, move them to the output directory
                os.makedirs(os.path.join(out_base_path, f"{package_name}-{latest_version}"))
                for obj in os.listdir(tmp_dir):
                    shutil.move(os.path.join(tmp_dir, obj), os.path.join(out_base_path, f"{package_name}-{latest_version}"))
            else:
                os.rename(os.path.join(tmp_dir, extracted_dirs[0]), os.path.join(tmp_dir, f"{package_name}-{latest_version}"))
                shutil.move(os.path.join(tmp_dir, f"{package_name}-{latest_version}"), out_base_path)

            return os.path.join(out_base_path, f"{package_name}-{latest_version}")
        except Exception as e:
            # print(f"Failed to extract {package_name}-{latest_version}.{package_ext}: {e}")
            return
        finally:
            # remove the temporary directory
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if os.path.exists(pkg_path):
                os.remove(pkg_path)


def run_scan(packages_db_path, packges_save_path, dataset_path, malware_found_path):
    packages_db_path = os.path.join(OUT_PATH, packages_db_path)
    packages_save_path = os.path.join(DATA_BASE_PATH, packges_save_path)
    dataset_path = os.path.join(OUT_PATH, dataset_path)
    malware_found_path = os.path.join(OUT_PATH, malware_found_path)

    scan_new_packages(packages_db_path, packages_save_path, dataset_path, malware_found_path)

def run_scan_live2():
    run_scan('db_packages_pypi_live2.json', 'dataset_pypi_live2', 'dataset_pypi_live2.csv', 'malware_live2.txt')


def run_scan_new():
    run_scan('db_packages_pypi_new_vetting.json', 'dataset_pypi_new_vetting', 'dataset_pypi_new_vetting.csv', 'malware_new_vetting.txt')


if __name__ == "__main__":
    # run_scan_live2()
    run_scan_new()