from contextlib import redirect_stderr
import abc
import os
import joblib
from feature_extractor import extract_features_package
from feature_extractor_soa_detector import extract_features_package as extract_soa_features_package
import numpy as np
from utils import Package
# from guarddog.analyzer.sourcecode import get_sourcecode_rules, SempgrepRule, YaraRule
# from guarddog.ecosystems import ECOSYSTEM
from guarddog.scanners import PypiPackageScanner, NPMPackageScanner


class Model(metaclass=abc.ABCMeta):
    """Abstract machine learning model wrapper."""

    @abc.abstractmethod
    def extract_features(self, sample):
        """
        Extracts a feature vector from the input object.

        Arguments:
            sample : An input point that belongs to the input space of the wrapped model.

        Returns:
            feature_vector (numpy ndarray) : array containing the feature vector of the input sample.

        Raises:
            NotImplementedError: this method needs to be implemented
        """
        raise NotImplementedError("extract_features not implemented in abstract class")

    @abc.abstractmethod
    def classify(self, sample):
        """
        Returns the probability of belonging to a particular class.
        It calls the extract_features function on the input value to produce a feature vector.

        Arguments:
            sample : Input object

        Returns:
            float : the probability of belonging to the malicious class
        """
        raise NotImplementedError("classify not implemented in abstract class")


class SklearnModel(Model):
    """Scikit learn classifier wrapper class"""

    def __init__(self, model_path, use_soa_features=False):
        """
        Constructs a wrapper around an scikit-learn classifier, or equivalent.
        It must implement predict_proba function.

        Arguments:
            model_path: path of the ML model
            use_soa_features: if True, uses the SoA features else the default ones

        Raises:
            Exception:
                - not implement predict_proba
                - cannot load the ML model
        """
        try:
            sklearn_classifier = joblib.load(model_path)
            # disable parallel processing to avoid issues with the parallelization of the fuzzer
            if hasattr(sklearn_classifier, "n_jobs"):
                sklearn_classifier.n_jobs = 1
        except Exception:
            raise Exception("Error in loading model.")

        if getattr(sklearn_classifier, "predict_proba", None) is None:
            raise Exception("object does not implement predict_proba function")
        
        self._feat_extractor = extract_soa_features_package if use_soa_features else extract_features_package

        self._model = sklearn_classifier

    def classify(self, sample: Package):
        """
        Returns the probability of belonging to the positive class (i.e., malware).
        It calls the extract_features function on the input value to produce a feature vector.

        Arguments:
            sample: an input sample represented by an instance of Package class.

        Returns:
            float: the confidence score of the input sample.
        """
        feature_vector = self.extract_features(sample)
        feature_vector = feature_vector.reshape(1, -1)
        y_pred = self._model.predict_proba(feature_vector)
        return y_pred[0, 1]

    def extract_features(self, sample):
        """
        Returns the feature representation of the input.

        Arguments:
            sample: an input sample represented by an instance of Package class.

        Returns:
            numpy ndarray: the vector representation of the input sample.
        """
        return np.array(self._feat_extractor(sample))


class GuarddogWrapper(Model):
    """Datadog GuardDog wrapper class"""

    def __init__(self, ecosystem='pypi'):
        """
        Constructs a wrapper for the Datadog GuardDog

        Raises:
            Exception:
                - cannot run GuardDog (not installed)
        """
        try:
            import guarddog
        except ImportError:
            raise Exception("GuardDog is not installed. Plase install it using 'pip install guarddog'")
        
        assert ecosystem in ['pypi', 'npm'], f"Unsupported ecosystem: {ecosystem}"
        self.ecosystem = ecosystem

    def classify(self, sample: Package):
        """
        Returns the probability of belonging to the positive class (i.e., malware).
        It calls the extract_features function on the input value to produce a feature vector.

        Arguments:
            sample: an input sample represented by an instance of Package class.

        Returns:
            float: the confidence score of the input sample.
        """
        results = self.extract_features(sample)
        # return results['issues'] / len(results['results'])
        return results['issues']

    def extract_features(self, sample):
        """
        Returns the feature representation of the input.

        Arguments:
            sample: an input sample represented by an instance of Package class.

        Returns:
            sample: the input sample
        """
        if self.ecosystem == 'pypi':
            scanner = PypiPackageScanner()
        elif self.ecosystem == 'npm':
            scanner = NPMPackageScanner()

        results = scanner.scan_local(sample.path)
        return results