# One Detector Fits All: Robust and Adaptive Detection of Malicious Packages from PyPI to Enterprises (ACSAC '25)

This repository contains the materials related to the paper _One Detector Fits All: Robust and Adaptive Detection of Malicious Packages from PyPI to Enterprises_ accepted
at the [41st Annual Computer Security Applications Conference (ACSAC) (ACSAC '25)](https://www.acsac.org/2025/).  


## Organization
This repository is organized as follows:

**repository root folder**  
It includes the following files:  

* `install.sh`: script to install the required Python packages (see `requirements.txt`).
* `LICENSE`: license file (`Apache 2.0`).
* `README.md` (this file): README file that describes the structure of this repository.

**`artifact/` folder**  
It includes the files related to the artifact.
Specifically, it is organized as follows:

* **`code/` folder**  
It contains the all the source code used in the experiments, organized as follow:
  * `run_training.py`: Python script to train the detectors evaluated in this work. It supports the following run modes:
    - `train-base`: train the models on MalwareBench using stratified 5-fold Cross Validation.
    - `train-full`: train the models on the whole MalwareBench without using stratified 5-fold CV. The trained models are then evaluated on the real-world datasets (`live1` and `live2`).
    - `adv-train`: train the XGBoost model using adversarial training (AT) using stratified 5-fold CV.
    - `adv-train-full`: train the XGBoost model using AT using the whole training (MalwareBench) and adversarial training sets. The default percentage of samples selected from the adversarial training set is 20%.
    - `adv-train-full-perc`: same as `adv-train-full` but it allows to evaluate several percentages of AT samples (10, 20, 30, 40, 50, 60, 70, 80, 90, 100).
    - `train-final-detector-full`: train the XGBoost model on MalwareBench and `live1` datasets.
    - `adv-train-final-detector-full`: same as `train-final-detector-full` but trains the XGBoost model using AT: the XGBoost model is trained on MalwareBench, `live1` and the full adversarial training datasets.

  * `run_experiments.py`: it contains the source code to run the real-world experiments and the two case studies (PyPI and industrial scenarios).  
    It supports the following run modes:
    - `advtrain-perc`: compute the results related to Adversarial Training experiments (see Table 4 in the paper).
    - `real-world-live1`: compute the results related to the real-world experiments based on `live1` dataset.
    - `real-world-live2`: compute the results related to the real-world experiments based on `live2` dataset.
    - `pypi-case-study`: compute the results related to the PyPI case study  
    - `enterprise-case-study`: used to compute the results related to the enterprise case study.
  
  * `analyze_results_malwarebench.ipynb`: Jupyter notebook to generate the results related to the baseline evaluation and the ROC curves of the detectors evaluated on MalwareBench (see Figure 6 in the paper).

  * `run_guarddog.py`: Python script to evaluate GuardDog

  * `run_vetting.py`: Python script to use the final model for real-time vetting of PyPI packages.

  * `feature_extractor.py`: it contains the source code to extract the features proposed by [Ladisa et al.](https://dl.acm.org/doi/abs/10.1145/3627106.3627138).

  * `feature_extractor_soa_detector.py`: it contains the source code to extract the SoA features proposed in our work (see `FeatureExtractorSoaDetector` class), which extend the features proposed by Ladisa et al.

  * `features_api_behavior_soa_detector.py`: it contains several functions used by the `FeatureExtractorSoaDetector` class to extract the API- and behavior-related features.

  * `features_obfuscation_soa_detector.py`: it contains several functions used by the `FeatureExtractorSoaDetector` class to extract the obfuscation-related features based on the adversarial transformations proposed in this work.

  * `transformations.py`: it contains the source code of the proposed adversarial source code transformations.

  * `optimizer.py`: it contains the source code of the black-box optimization algorithm used to optimize the transformations against a target detector.

  * `model.py`: it implements a wrapper [Scikit-learn](https://scikit-learn.org/stable/) models and [GuardDog](https://github.com/DataDog/guarddog) to be used with the black-box optimization algorithm.

  * `run_attack.py`: Python script to generate adversarial packages for a target detector to evaluate its robustness or for AT.

  * `run_experiments_adv.py`: main Python script to run the experiments to evaluate the robustness of the target detectors (i.e., generate the adversarial packages for the 
    `adversarial test set`) and to generate the adversarial packages for the `adversarial training set` to be used for AT.
    It supports the following run modes:
    - `test`: generate the adversarial packages from the `test set` to evaluate the robustness of all the detectors.
    - `adv-train`: generate the adversarial packages from the `training set` for AT of the XGBoost-based detector.
    - `test-retrained`: generate the adversarial packages from the `test set` to evaluate the robustness of the AT-based XGBoost detector.

  * `utils.py`: it includes some utility functions used in the some of the files above.

* **`data/` folder**
It contains the list of malware found in the vetting related to `live1` (`malware_live1.txt`), as well as the `malware_live2_info.csv` file including the information about the malware found in the PyPi vetting case study. The latter is also used by the `analysis_malware_live2.ipynb` Jupyter notebook to generate the results for Tables 7-9 (see `claims/` folder).

* **`out/` folder**  
It contains an example of dataset used for the evaluation, specifically the dataset of SoA features extracted from MalwareBench (`dataset_pypi_malwarebench_soa_features.csv`).

* **`models/` folder**  
It contains the final detector trained using AT on MalwareBench, `live1` and the `adversarial training set` (generated from MalwareBench).
It can be used in conjunction with `run_vetting.py` for real-time vetting of PyPI packages.

**`claims/` folder**  
In this directory we provide the key results of the experiments including the Tables 3-10 and ROC curves of the detectors evaluated on MalwareBench dataset (see Figure 6).


## Requirements

### Hardware dependencies
We conducted our experimental evaluation on a server equipped with an Intel Xeon Platinum 8160 CPU @ 2.10 GHz (64 cores) and 256 GB of RAM.
No GPU is needed to train the evaluated machine-learning models.

### Software dependencies
The experiments have been validated on a server based on Ubuntu 22.04.6 LTS.  
The source code has been tested using Python 3.10.12. All the required packages are provided in the `requirements.txt` file included in the root folder of this repository.

### Data dependencies
The real-world experiemnts and case studies assume to have the `live1`, `live2` and the data related to the enetrprise case study available in the `data/` folder.
However, for now they cannot be shared, but we provide the list of malware packages found in the real-world and PyPI case study experiments.

## Instructions

1. Retrive the artifact:  
`git clone https://github.com/SAP-samples/robust-pypi-detector.git`

2. Move into the artifact root folder:  
`cd robust-pypi-detector/artifact`

3. Run install script to create and activate a new Python virtual environment:  
`./install.sh`  
`source advpkg_env/bin/activate`

4. Get the MalwareBench dataset from GitHub:  
`git clone https://github.com/MalwareBench/pypi.git`  
**NOTE:** It is a private repository, hence you need to request access to it if needed.

5. Move to the source code folder:  
`cd ./code`

6. Process the MalwareBench dataset:  
`python process_malwarebench.py`

7. Train the target ML models on MalwareBench:  
`python run_training.py train-base --soa-features`  
`python run_training.py train-full --soa-features`

8. Evaluate GuardDog on MalwareBench:  
`python run_guarddog.py`

9. Generate the adversarial packages from MalwareBench to evaluate the robustness of the target ML models:  
`python run_experiments_adv.py test --soa-features`

10. Generate the adversarial packages from MalwareBench for AT:  
`python run_experiments_adv.py adv-train --soa-features`

11. Train the target XGBoost models using AT:  
`python run_training.py adv-train --soa-features`  
`python run_training.py adv-train-full --soa-features`  
`python run_training.py adv-train-full-perc --soa-features`

12. Test the robustness of the AT-based XGBoost detectors:  
`python run_experiments_adv.py test-retrained --soa-features`

13. Generate the results of the baseline evaluation and plot the ROC by running the `analyze_results_malwarebench.ipynb` Jupyter notebook.

14. Evaluate the AT-based models on `live1`:  
`python run_experiments.py advtrain-perc`

15. Train the final detector on MalwareBench and `live1`:  
`python run_training.py train-final-detector-full --soa-features`  
`python run_training.py adv-train-final-detector-full --soa-features`  

16. Generate results for the real-world experiments based on `live1` and `live2`:  
`python run_experiments.py real-world-live1`  
`python run_experiments.py real-world-live2`

17. Generate results for the PyPI case study:  
`python run_experiments.py pypi-case-study`

18. Generate results for the enterprise case study:  
`python run_experiments.py enterprise-case-study`

19. Generate and print the results related to the analysis of malware in `live2` by running the `analysis_malware_live2.ipynb` Jupyter notebook.  
It generates the related tables (Tables 7-9) in the `claims/` folder.

20. Run a new vetting using the final model:  
`python run_vetting.py`

## How to obtain support
[Create an issue](https://github.com/SAP-samples//issues) in this repository if you find a bug or have questions about the content.

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2024 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file.
