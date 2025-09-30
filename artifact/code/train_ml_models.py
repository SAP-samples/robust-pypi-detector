from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.utils import shuffle
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import json
import pandas as pd
import os
import joblib
from utils import plot_roc
import numpy as np
import xgboost
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


ML_MODELS = ['dt', 'rf', 'xgboost']
ML_MODELS_ADV = ['xgboost']
NUM_FOLDS = 5

OUT_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/out'
EVAL_RESULTS_PATH = os.path.join(OUT_PATH, 'eval_results_malwarebench/results_')


def get_model_params(model_type):
    if model_type == 'dt':
        model = DecisionTreeClassifier(class_weight="balanced", random_state=123)
        model_params = {
            'max_depth': Integer(2, 4),
            'max_features': Categorical(['sqrt', 'log2', None]), 
            'criterion': Categorical(['gini', 'entropy', 'log_loss']),
            'min_samples_leaf': Integer(4, 8),
            'min_samples_split': Integer(6, 16)
        }
    elif model_type == 'rf':
        model = RandomForestClassifier(class_weight="balanced", n_jobs=8, random_state=123)
        model_params = {
            'max_depth': Integer(2, 4),
            'max_features': Categorical(['sqrt', 'log2', None]), 
            'n_estimators': Integer(64, 256), 
            'criterion': Categorical(['gini', 'entropy', 'log_loss']),
            'min_samples_leaf': Integer(4, 8),
            'min_samples_split': Integer(6, 16),
            'max_samples': Real(0.1, 1)
        }
    elif model_type == 'xgboost':
        # model = GradientBoostingClassifier(class_weight="balanced")
        model = xgboost.XGBClassifier(n_jobs=8, random_state=123)
        model_params = {
            'max_depth': Integer(2, 4),
            'n_estimators': Integer(64, 256),
            'min_child_weight': Integer(8, 16),
            'gamma': Real(0.6, 1.2),
            'eta': Real(0.08, 0.16),
            'colsample_bytree': Real(0.1, 0.3)
        }

    return model, model_params


def train_ml_models(models, dataset_path, num_folds, out_path, labels_split=None, rand_state=0, log_results=True):

    assert os.path.isfile(dataset_path)
    assert isinstance(num_folds, int) and num_folds > 0
    assert labels_split is None or isinstance(labels_split, list)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # load the dataframe from CSV
    dataset = pd.read_csv(dataset_path)
    X, y = dataset.iloc[:,1:-1].to_numpy(dtype='float'), dataset.iloc[:,-1].to_numpy()
    is_multiclass = len(np.unique(y)) > 2

    cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=rand_state)
    target_labels = y if labels_split is None else labels_split
    for fold_idx, (train_index, test_index) in enumerate(cv.split(X, target_labels)):
        X_train_val, y_train_val = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        with open(os.path.join(out_path, f'test_indices_fold{fold_idx+1}.json'), 'w') as file:
            json.dump(test_index.tolist(), file)

        for model_type in models:
            model, model_params = get_model_params(model_type)
            if log_results:
                logger = open(os.path.join(out_path, f'results_{model_type}.txt'), 'a+')
            
            # clf = GridSearchCV(estimator=model, param_grid=model_params, cv=num_folds, n_jobs=-1)
            # n_jobs = len(os.sched_getaffinity(0)) - 1
            n_jobs = 40
            bayes_cv = BayesSearchCV(estimator=model, search_spaces=model_params, n_iter=30, cv=num_folds,
                                     scoring='f1_weighted' if is_multiclass else 'f1', n_jobs=n_jobs, random_state=rand_state)
            bayes_cv.fit(X_train_val, y_train_val)
            
            best_model = bayes_cv.best_estimator_
            y_scores = best_model.predict_proba(X_test)[:, 1]
            y_pred = best_model.predict(X_test)
            f1_value = f1_score(y_test, y_pred, average='weighted' if is_multiclass else 'binary')
            if log_results:
                logger.write(f"F1-score {model_type} fold #{fold_idx+1}: {f1_value}\n")
                logger.close()
            else:
                print(f"F1-score {model_type} fold #{fold_idx+1}: {f1_value}")
            out_save_path = os.path.join(out_path, f'result_{fold_idx+1}')
            os.makedirs(out_save_path, exist_ok=True)
            if not is_multiclass:
                plot_roc(y_test, y_scores, label=f'{model_type} fold #{fold_idx+1}', plot_rand_guessing=False, log_scale=True,
                            save_path=os.path.join(out_save_path, f'roc_{model_type}_fold{fold_idx+1}.pdf'))

            joblib.dump(bayes_cv.best_estimator_, os.path.join(out_save_path, f'{model_type}_fold{fold_idx+1}.joblib'))
            # print()


def advtrain_ml_models(models, dataset_path, adv_datasets_base_path, num_folds, train_base_path, out_path, rand_state=0, log_results=True, perc_adv_samples_fold=None, sort_by_score=False, eval_results_base_path=None):

    assert os.path.isfile(dataset_path)
    assert isinstance(num_folds, int) and num_folds > 0
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # load the dataframe from CSV
    dataset = pd.read_csv(dataset_path)
    X, y = dataset.iloc[:,1:-1].to_numpy(dtype='float'), dataset.iloc[:,-1].to_numpy()
    is_multiclass = len(np.unique(y)) > 2

    for fold_idx in range(num_folds):
        with open(os.path.join(train_base_path, f'test_indices_fold{fold_idx+1}.json'), 'r') as file:
            test_index = json.load(file)

        train_index = [idx for idx in range(X.shape[0]) if idx not in test_index]
        X_train_val, y_train_val = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        for model_type in models:
            model, model_params = get_model_params(model_type)
            if log_results:
                logger = open(os.path.join(out_path, f'results_{model_type}.txt'), 'a+')

            ##### ADV SAMPLES ####
            adv_dataset_path = os.path.join(adv_datasets_base_path.format(model=model_type), f"dataset_advtrain_{model_type}_fold{fold_idx+1}.csv")
            assert os.path.isfile(adv_dataset_path)
            # print(f"Loading adversarial samples from {adv_dataset_path}")
            dataset_adv = pd.read_csv(adv_dataset_path)
            X_adv, y_adv = dataset_adv.iloc[:,1:-1].to_numpy(dtype='float'), dataset_adv.iloc[:,-1].to_numpy()

            if sort_by_score:
                # compute indices to sort the adversarial samples by the score of the model (from lowest, most adversarial, to highest, less adversarial)
                assert eval_results_base_path is not None
                # print(f"Using eval results from {eval_results_base_path.format(model=model_type, fold_idx=fold_idx+1)}")
                with open(eval_results_base_path.format(model=model_type, fold_idx=fold_idx+1), 'r') as fp:
                    eval_results = fp.readlines()

                pkg_scores = {}
                for result in eval_results:
                    data = json.loads(result)
                    pkg_scores[data["sample"]] = data['best_score']
                    
                scores = []
                for pkg in dataset_adv['Package Name'].values:
                    scores.append(pkg_scores[pkg])

                sorted_idx = np.argsort(scores)

            # Random sample num_adv_samples_fold samples from each fold
            if perc_adv_samples_fold is not None:
                num_samples_adv_fold = int(X_adv.shape[0] * perc_adv_samples_fold)
                if sort_by_score:
                    # select the first num_samples_adv_fold samples from the sorted list
                    X_adv = X_adv[sorted_idx[:num_samples_adv_fold]]
                    y_adv = y_adv[sorted_idx[:num_samples_adv_fold]]
                else:
                    X_adv, y_adv = shuffle(X_adv, y_adv, n_samples=num_samples_adv_fold, random_state=rand_state)

            X_train_val_fold = np.vstack((X_train_val, X_adv))
            y_train_val_fold = np.concatenate((y_train_val, y_adv))
            #####################

            # clf = GridSearchCV(estimator=model, param_grid=model_params, cv=num_folds, n_jobs=-1)
            # n_jobs = len(os.sched_getaffinity(0)) - 1
            n_jobs = 8
            bayes_cv = BayesSearchCV(estimator=model, search_spaces=model_params, n_iter=30, cv=num_folds,
                                     scoring='f1_weighted' if is_multiclass else 'f1', n_jobs=n_jobs, random_state=rand_state)
            bayes_cv.fit(X_train_val_fold, y_train_val_fold)
            best_model = bayes_cv.best_estimator_
            y_scores = best_model.predict_proba(X_test)[:, 1]
            y_pred = best_model.predict(X_test)
            f1_value = f1_score(y_test, y_pred, average='weighted' if is_multiclass else 'binary')
            if log_results:
                logger.write(f"F1-score {model_type} fold #{fold_idx+1}: {f1_value}\n")
                logger.close()
            else:
                print(f"F1-score {model_type} fold #{fold_idx+1}: {f1_value}")
            out_save_path = os.path.join(out_path, f'result_{fold_idx+1}')
            os.makedirs(out_save_path, exist_ok=True)
            if not is_multiclass:
                plot_roc(y_test, y_scores, label=f'{model_type} fold #{fold_idx+1}', plot_rand_guessing=False, log_scale=True,
                         save_path=os.path.join(out_save_path, f'roc_{model_type}_fold{fold_idx+1}.pdf'))

            joblib.dump(bayes_cv.best_estimator_, os.path.join(out_save_path, f'{model_type}_fold{fold_idx+1}.joblib'))
            # print()
            # logger.close()


def train_ml_models_full_dataset(models, dataset_path, out_path, dataset_test_path=None, rand_state=0, log_results=True):

    assert os.path.isfile(dataset_path)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # load the dataframe from CSV
    dataset = pd.read_csv(dataset_path)

    X, y = dataset.iloc[:,1:-1].to_numpy(dtype='float'), dataset.iloc[:,-1].to_numpy()

    if dataset_test_path is not None:
        assert os.path.isfile(dataset_test_path)
        dataset_test = pd.read_csv(dataset_test_path)
        assert 'label' == dataset_test.columns[-1]
        X_test, y_test = dataset_test.iloc[:,1:-1].to_numpy(dtype='float'), dataset_test.iloc[:,-1].to_numpy()
    else:
        X_test, y_test = X, y

    is_multiclass = len(np.unique(y)) > 2

    for model_type in models:
        model, model_params = get_model_params(model_type)
        if log_results:
            logger = open(os.path.join(out_path, f'results_{model_type}_final.txt'), 'a+')

        # clf = GridSearchCV(estimator=model, param_grid=model_params, cv=num_folds, n_jobs=-1)
        n_jobs = len(os.sched_getaffinity(0)) - 1
        bayes_cv = BayesSearchCV(estimator=model, search_spaces=model_params, n_iter=30, cv=5,
                                 scoring='f1_weighted' if is_multiclass else 'f1', n_jobs=n_jobs, random_state=rand_state)
        bayes_cv.fit(X, y)
        best_model = bayes_cv.best_estimator_

        y_scores = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)
        f1_value = f1_score(y_test, y_pred, average='weighted' if is_multiclass else 'binary')
        recall_value = recall_score(y_test, y_pred, average='weighted' if is_multiclass else 'binary')
        precision_value = precision_score(y_test, y_pred, average='weighted' if is_multiclass else 'binary')
        if log_results:
            logger.write(f"F1-score {model_type}: {f1_value}\n")
            logger.close()
        else:
            print(f"F1-score {model_type}: {f1_value}")

        if not is_multiclass:
            plot_roc(y_test, y_scores, label=f'{model_type}', plot_rand_guessing=False, log_scale=True,
                     save_path=os.path.join(out_path, f'roc_{model_type}_final.pdf'))

        joblib.dump(best_model, os.path.join(out_path, f'{model_type}_final.joblib'))


def advtrain_ml_models_full_dataset(models, dataset_path, adv_datasets_base_path, out_path, dataset_test_path=None, perc_adv_samples_fold=None,
                                    rand_state=0, log_results=True, sort_by_score=False, eval_results_base_path=None):
    assert os.path.isfile(dataset_path)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # load the dataframe from CSV
    dataset = pd.read_csv(dataset_path)

    X, y = dataset.iloc[:,1:-1].to_numpy(dtype='float'), dataset.iloc[:,-1].to_numpy()

    if dataset_test_path is not None:
        assert os.path.isfile(dataset_test_path)
        test_dataset = pd.read_csv(dataset_test_path)
        X_test, y_test = test_dataset.iloc[:,1:-1].to_numpy(dtype='float'), test_dataset.iloc[:,-1].to_numpy()
    else:
        X_test, y_test = X, y
    
    is_multiclass = len(np.unique(y)) > 2

    for model_type in models:
        model, model_params = get_model_params(model_type)
        if log_results:
            logger = open(os.path.join(out_path, f'results_{model_type}_full.txt'), 'a+')

        X_adv_full, y_adv_full = np.array([]), np.array([])
        for fold_idx in range(NUM_FOLDS):
            adv_dataset_path = os.path.join(adv_datasets_base_path.format(model=model_type), f"dataset_advtrain_{model_type}_fold{fold_idx+1}.csv")
            assert os.path.isfile(adv_dataset_path)
            dataset_adv = pd.read_csv(adv_dataset_path)
            X_adv, y_adv = dataset_adv.iloc[:,1:-1].to_numpy(dtype='float'), dataset_adv.iloc[:,-1].to_numpy()

            if sort_by_score:
                assert eval_results_base_path is not None
                # sort the adversarial samples by the score of the model (from lowest, most adversarial, to highest)
                with open(eval_results_base_path.format(model=model_type, fold_idx=fold_idx+1), 'r') as fp:
                    eval_results = fp.readlines()

                pkg_scores = {}
                for result in eval_results:
                    data = json.loads(result)
                    pkg_scores[data["sample"]] = data['best_score']
                    
                scores = []
                for pkg in dataset_adv['Package Name'].values:
                    scores.append(pkg_scores[pkg])

                sorted_idx = np.argsort(scores)

            # Random sample num_adv_samples_fold samples from each fold
            if perc_adv_samples_fold is not None:
                num_samples_adv_fold = int(X_adv.shape[0] * perc_adv_samples_fold)
                if sort_by_score:
                    # select the first num_samples_adv_fold samples from the sorted list
                    X_adv = X_adv[sorted_idx[:num_samples_adv_fold]]
                    y_adv = y_adv[sorted_idx[:num_samples_adv_fold]]
                else:
                    # shuffle the samples and select the first num_samples_adv_fold.
                    # This ensures reproducibility and samples are randomly selected in an incremental way
                    # >>> from sklearn.utils import shuffle
                    # >>> a = list(range(10))
                    # >>> shuffle(a, random_state=0)
                    # [2, 8, 4, 9, 1, 6, 7, 3, 0, 5]
                    # >>> shuffle(a, random_state=0)   # random state ensures reproducibility
                    # [2, 8, 4, 9, 1, 6, 7, 3, 0, 5]
                    # >>> shuffle(a, n_samples=3, random_state=0)  # select first 3 samples from the shuffled list
                    # [2, 8, 4]
                    # >>> shuffle(a, n_samples=5, random_state=0)  # select first 5 samples from the shuffled list
                    # [2, 8, 4, 9, 1]
                    # >>> shuffle(a, n_samples=8, random_state=0)  # select first 8 samples from the shuffled list
                    # [2, 8, 4, 9, 1, 6, 7, 3]
                    X_adv, y_adv = shuffle(X_adv, y_adv, n_samples=num_samples_adv_fold, random_state=rand_state)

            X_adv_full = np.vstack((X_adv_full, X_adv)) if X_adv_full.size else X_adv
            y_adv_full = np.concatenate((y_adv_full, y_adv)) if y_adv_full.size else y_adv

        X_train_model = np.vstack((X, X_adv_full))
        y_train_model = np.concatenate((y, y_adv_full))

        # clf = GridSearchCV(estimator=model, param_grid=model_params, cv=num_folds, n_jobs=-1)
        n_jobs = len(os.sched_getaffinity(0)) - 1
        bayes_cv = BayesSearchCV(estimator=model, search_spaces=model_params, n_iter=30, cv=5,
                                 scoring='f1_weighted' if is_multiclass else 'f1', n_jobs=n_jobs, random_state=rand_state)
        bayes_cv.fit(X_train_model, y_train_model)
        best_model = bayes_cv.best_estimator_

        y_scores = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)
        f1_value = f1_score(y_test, y_pred, average='weighted' if is_multiclass else 'binary')
        recall_value = recall_score(y_test, y_pred, average='weighted' if is_multiclass else 'binary')
        precision_value = precision_score(y_test, y_pred, average='weighted' if is_multiclass else 'binary')
        if log_results:
            logger.write(f"F1-score {model_type}: {f1_value}\n")
            logger.close()
        else:
            print(f"F1-score {model_type}: {f1_value}")

        if not is_multiclass:
            plot_roc(y_test, y_scores, label=f'{model_type}', plot_rand_guessing=False, log_scale=True,
                     save_path=os.path.join(out_path, f'roc_{model_type}_final.pdf'))

        perc_str = 100 if perc_adv_samples_fold is None else int(perc_adv_samples_fold*100)
        model_savename = f'{model_type}_final_{perc_str}.joblib'
        joblib.dump(best_model, os.path.join(out_path, model_savename))
