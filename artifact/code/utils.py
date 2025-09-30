import base64
import glob
import json
import numpy as np
import os
import random
import requests
import shutil
import string
import tarfile
import zipfile
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import shutil
import pandas as pd
import re
import subprocess


class Package:
    def __init__(self, name, version, pkg_type, path, url=None) -> None:
        assert isinstance(name, str)
        assert isinstance(version, str)
        assert isinstance(pkg_type, str) and pkg_type in ['pypi', 'npm'], f"Unsupported package type: {pkg_type}"
        assert isinstance(path, str) and os.path.isdir(path)
        assert isinstance(url, str) or url is None

        self.name = name
        self.version = version
        self.path = path
        self.url = url
        self.type = pkg_type

        if self.type == 'pypi':
            self.metadata_path = [os.path.join(path, 'setup.py')] if os.path.isfile(os.path.join(path, 'setup.py')) else []
            self.code_path = [str(file_path) for file_path in Path(self.path).rglob('*.py')
                              if (file_path.is_file() and not str(file_path).endswith('setup.py'))]
        elif self.type == 'npm':
            self.metadata_path = [os.path.join(path, 'package.json')] if os.path.isfile(os.path.join(path, 'package.json')) else []
            self.code_path = [str(file_path) for file_path in Path(self.path).rglob('*.js') if file_path.is_file()]

    def clone(self, out_path):
        shutil.copytree(self.path, out_path)
        return Package(self.name, self.version, self.type, out_path, self.url)


def random_char():
    """
    Returns a random character.
    
    Raises:
        AssertionError: bad type passed as argument

    Returns:
        str : random character
    """

    chars = list(string.digits + string.ascii_letters)
    return random.choice(chars)


def random_string(max_len=10):
    """
    Creates a random string.

    Arguments:
        max_length (int) : the maximum length of the string [default=5]
        spaces (bool) : if True, all the printable character will be considered. Else, only letters and digits [default=True]

    Raises:
        TypeError: bad type passed as argument

    Returns:
        (str) : random string

    """
    assert isinstance(max_len, int)
    rand_s = random.choice(list(string.ascii_letters))

    return rand_s + "".join([random_char() for _ in range(random.randint(1, max_len-1))])


def check_base64(string):
    try:
        # Decode the string
        decoded_data = base64.b64decode(string, validate=True)
        # Check if the decoded string can be encoded back to the original string
        return base64.b64encode(decoded_data).decode('utf-8') == string
    except Exception:
        # An exception occurred during decoding, indicating that the string is not base64-encoded
        return False


def get_stats_tokens(num_tokens_file):
    assert isinstance(num_tokens_file, list)

    if not num_tokens_file:
        return 0.0, 0.0, 0.0, 0.0
    return np.mean(num_tokens_file), np.max(num_tokens_file), np.std(num_tokens_file), np.quantile(num_tokens_file, 0.75)


# generalization languages
def gen_language_3(value):
    pattern = ''
    value = list(str(value))
    for c in value:
        if c.isnumeric():
            pattern += 'd'
        elif c.isalpha():
            pattern += 'l'
        else:
            pattern += 's'
    
    # grouped_pattern = [''.join(g) for _, g in groupby(pattern)]
    # ''.join([f'{v[0]}({len(v)})' for v in grouped_pattern])
    return (pattern)

def gen_language_4(value):
    pattern = ''
    value = list(str(value))
    for c in value:
        if c.isnumeric():
            pattern += 'd'
        elif c.isupper():
            pattern += 'u'
        elif c.islower():
            pattern +='l'
        else:
            pattern += 's'
    
    # grouped_pattern = [''.join(g) for _, g in groupby(pattern)]
    # ''.join([f'{v[0]}({len(v)})' for v in grouped_pattern])
    return (pattern)

def gen_language_8(value):
    pattern = ''
    value = list(str(value))
    for c in value:
        if c.isnumeric():
            pattern += 'd'
        elif c.isupper():
            pattern += 'u'
        elif c.islower():
            pattern +='l'
        elif c=='.':
            pattern +='p'
        elif c=='/':
            pattern +='h'
        elif c=='-':
            pattern +='a' 
        elif c=='|' or c=='%' or c=='$'or c=='~'or c=='?':
            pattern +='i'
        else:
            pattern += 's'
    
    # grouped_pattern = [''.join(g) for _, g in groupby(pattern)]
    # ''.join([f'{v[0]}({len(v)})' for v in grouped_pattern])
    return (pattern)

def gen_language_16(value):
    pattern = ''
    value = list(str(value))
    for c in value:
        if c.isnumeric():
            pattern += 'd'
        elif c.isupper():
            pattern += 'u'
        elif c.islower():
            pattern +='l'
        elif c=='.':
            pattern +='p'
        elif c=='/':
            pattern +='h'
        elif c=='-':
            pattern +='a'
        elif c=='%':
            pattern +='p'
        elif c=='|':
            pattern +='i'
        elif c=='=':
            pattern +='e'
        elif c==':':
            pattern +='c'
        elif c=='$':
            pattern +='m'
        elif c=='>':
            pattern +='g'
        elif c=='<':
            pattern +='o'
        elif c=='~':
            pattern +='t'
        elif c=='?':
            pattern +='q'
        else:
            pattern += 's'
    
    # grouped_pattern = [''.join(g) for _, g in groupby(pattern)]
    # ''.join([f'{v[0]}({len(v)})' for v in grouped_pattern])
    return (pattern)


# shannon entropy function 
def shannon_entropy(data, base=2):
    entropy = 0.0
    if len(data) > 0:
        cnt = Counter(data)
        length = len(data)
        for count in cnt.values():
            entropy += (count / length) * -np.log2(count / length)
    return entropy


def check(s, arr):
    result = []
    for i in arr:
        # for every character in char array
        # if it is present in string return true else false
        if i in s:
            result.append("True")
        else:
            result.append("False")
    return result


# input list of identifiers transformed by the generalization language 
def get_num_heterogeneous_tokens(list_words, symbols=['u','d','l','s']):
    unique_symbols_id = []
    # get unique symbols from each identifier
    for id in list_words:
        unique_symbols_id.append("".join(set(id)))
    # initialize the count for obfuscation:
    counter_obfuscated = 0
    for id in unique_symbols_id:
        # upper case , digit, lower case, symbol 
        if (check(id, symbols)) == ['True', 'True', 'True', 'True']:
            counter_obfuscated += 1
        # upper case, digit, symbol
        if (check(id, symbols)) == ['True', 'True', 'False', 'True']:
            counter_obfuscated += 1
        # digit, lower case, symbol
        if (check(id, symbols)) == ['False', 'True', 'True', 'True']:
            counter_obfuscated += 1
        # digit, symbol 
        if (check(id, symbols)) == ['False', 'True', 'False', 'True']:
            counter_obfuscated += 1   
    return counter_obfuscated


def plot_roc(y_true, y_scores, label, ax=None, settings=None, add_title=False, plot_rand_guessing=True,
             save_path=None, log_scale=True, add_label=True, legend_settings=None):

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if settings is not None and isinstance(settings, dict):
        ax_settings = settings.copy()
    else:
        ax_settings = dict(lw=2)
    if add_label:
        ax_settings['label'] = "{}, auc={:.2f}".format(label, roc_auc)
    ax.plot(fpr, tpr, **ax_settings)
    if plot_rand_guessing:
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    if log_scale:
        ax.set_xscale('log')
        ax.set_xlim([0.00008, 1.05])
    else:
        ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    if add_title:
        ax.set_title("ROC {}".format(label))
    if legend_settings is not None:
        # legend_settings = {'loc': 'lower right'}
        ax.legend(**legend_settings)
    ax.grid(True)

    if save_path is not None and fig is not None:
        plt.savefig(save_path, bbox_inches='tight')


def create_new_file_py(base_path):
    target_file = None
    for pkg_folder in glob.glob(os.path.join(base_path, "**/"), recursive=True):
        # for filename in filename_candidates:
        while True:
            filename = random_string() + '.py'
            file_path = os.path.join(pkg_folder, filename)
            if not os.path.isfile(file_path):
                # create file
                with open(file_path, 'w'): pass
                target_file = file_path
                break

    return target_file


def process_malwarebench_dataset(malwarebench_base_path, out_path_goodware, out_path_malware, max_goodware=None, max_malware=None):
    packages_info_path = os.path.join(malwarebench_base_path, 'pypi_package_info.csv')
    packages_info = pd.read_csv(packages_info_path)

    if not os.path.isdir(out_path_goodware):
        os.makedirs(out_path_goodware)
    if not os.path.isdir(out_path_malware):
        os.makedirs(out_path_malware)

    assert max_goodware is None or (isinstance(max_goodware, int) and max_goodware > 0)
    assert max_malware is None or (isinstance(max_malware, int) and max_malware > 0)

    # collect all the packages, take the latest version and extract the content
    packages_db = {}
    for name, version, artifact_type, path, label in zip(packages_info['name'], packages_info['version'], packages_info['artifact_id'], packages_info['group_ID'], packages_info['threat_type']):
        if artifact_type not in ['tar-gz', 'zip', 'py3-none-any-whl']:
            continue
        if name not in packages_db:
            packages_db[name] = dict()

        packages_db[name][version] = {'artifact_type': artifact_type, 'label': 1 if label == 'malware' else 0}

    # collect the names of the packages already in the output directory
    goodware_packages_already_saved = set([obj.rsplit('-', 1)[0] for obj in os.listdir(out_path_goodware) if os.path.isdir(os.path.join(out_path_goodware, obj))])
    malware_packages_already_saved = set([obj.rsplit('-', 1)[0] for obj in os.listdir(out_path_malware) if os.path.isdir(os.path.join(out_path_malware, obj))])

    counter_goodware = len(goodware_packages_already_saved)
    counter_malware = len(malware_packages_already_saved)

    # sort the versions and take the latest one
    for name in packages_db:
        latest_version = sorted(packages_db[name].keys())[-1]
        artifact_type = packages_db[name][latest_version]['artifact_type']
        package_path = os.path.join(malwarebench_base_path, 'packages', name, latest_version, artifact_type)
        if os.path.isdir(package_path):
            if packages_db[name][latest_version]['label'] == 0:
                if name not in goodware_packages_already_saved:
                    if max_goodware is not None and counter_goodware >= max_goodware:
                        print(f"Reached the maximum number of goodware packages: {max_goodware}")
                        continue
                    shutil.copytree(package_path, os.path.join(out_path_goodware, f"{name}-{latest_version}"))
                    print(f"Goodware package saved: {name}-{latest_version}")
                    counter_goodware += 1
                else:
                    print(f"Goodware package already saved: {name}-{latest_version}")
            else:
                if name not in malware_packages_already_saved:
                    if max_malware is not None and counter_malware >= max_malware:
                        print(f"Reached the maximum number of malware packages: {max_malware}")
                        continue
                    shutil.copytree(package_path, os.path.join(out_path_malware, f"{name}-{latest_version}"))
                    print(f"Malware package saved: {name}-{latest_version}")
                    counter_malware += 1
                else:
                    print(f"Malware package already saved: {name}-{latest_version}")
        else:
            print(f"[WARN] Missing package: {name}-{latest_version}")