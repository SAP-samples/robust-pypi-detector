import glob
from time import perf_counter
from utils import Package
import shutil
import os
import random
from transformations import ObfuscateIPs, ObfuscateIPsMetadata, ObfuscateURLs, ObfuscateURLsMetadata, ObfuscateInstallMetadata, \
    ObfuscateBase64Chunks, ObfuscateBase64ChunksMetadata, ObfuscateAPIsPy, ObfuscateAPIsPyMetadata, \
    InjectPlus, InjectPlusMetadata, InjectEq, InjectEqMetadata, InjectNewline, InjectNewlineMetadata, \
    InjectTab, InjectTabMetadata, InjectSpace, InjectSpaceMetadata, InjectBrackets, InjectBracketsMetadata, \
    InjectIP, InjectIPMetadata, InjectURL, InjectURLMetadata, InjectRandCode, InjectRandCodeMetadata, InjectNewFile

DATA_BASE_PATH = os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/data'
PACKAGES_TEMP_POOL = os.path.join(DATA_BASE_PATH, 'packages_pool')


class Optimizer:

    def __init__(self, ml_model, num_rounds, only_sr=False, pool_dir=PACKAGES_TEMP_POOL, save_only_best=False, debug=False):
        """Initialize an optimizer object.
        Arguments:
            ml_model: the input model to evaluate
            num_rounds (int): number of mutation rounds for the multi-round (MR) transformations.
            save_only_best (bool): flag to save the score only when it decreases.
            debug (bool): flag to enable verbose mode.

        Raises:
            AssertionError: num_rounds is not an integer
        """
        assert isinstance(num_rounds, int) and num_rounds > 0
        assert isinstance(save_only_best, bool)
        assert isinstance(debug, bool)

        self.model = ml_model
        self.num_rounds = num_rounds
        self.debug = debug
        self.save_only_best = save_only_best
        self.only_sr = only_sr

        self.best_score = None
        self.best_sample = None
        self.num_queries = 0
        self.traces = []
        self.pool_dir = pool_dir

    def _evaluate(self, candidates):
        """
        Evaluate a set of candidates (the mutated phishing webpages generated from the current best adversarial example),
        and uodate the best adversarial example found so far each time we found a better candidate.

        Arguments:
            candidates: a list of tuples having the following format: (html, url, transformation), where:
                        * html is a BeutifulSoup object
                        * url is the URL of the phishing webpage
                        * transformation is a string representing the used transformation
        """
        for package, transformation in candidates:
            score = self.model.classify(package)
            self.num_queries += 1

            if score <= self.best_score:
                self.best_score = score
                self.best_sample = package
                if self.save_only_best:
                    self.traces.append((self.num_queries, self.best_score, transformation))
                # if self.debug:
                #     print("Score after {} queries: {:.3f} using {}".format(self.num_queries, self.best_score, transformation))
            else:
                shutil.rmtree(package.path)

            if not self.save_only_best:
                self.traces.append((self.num_queries, score, transformation))

    def optimize(self, package: Package, out_log=None):
        """
        Generates an optimized adversarial package

        Arguments:
            package: input package (i.e., an object of Packsge class).

        Returns:
            tuple: a tuple representing the reult of the optimization:
                   * self.best_score: the best score
                   * self.best_sample: the final adversarial package
                   * self.num_queries: the number of used queries
                   * run_time: the run time in seconds
                   * self.traces: the optimization traces representing which transformation has been used in each query
        """
        lang = 'py' if package.type == 'pypi' else None

        multi_round_transformations = [
            InjectPlus(lang), InjectPlusMetadata(lang), InjectEq(lang), InjectEqMetadata(lang), InjectNewline(lang), InjectNewlineMetadata(lang),
            InjectTab(lang), InjectTabMetadata(lang), InjectSpace(lang), InjectSpaceMetadata(lang), InjectBrackets(lang), InjectBracketsMetadata(lang),
            InjectIP(lang), InjectIPMetadata(lang), InjectURL(lang), InjectURLMetadata(lang), InjectRandCode(lang), InjectRandCodeMetadata(lang), InjectNewFile(lang)
        ]

        single_round_transformations = [
            ObfuscateIPs(lang), ObfuscateIPsMetadata(lang), ObfuscateURLs(lang), ObfuscateURLsMetadata(lang), ObfuscateInstallMetadata(lang),
            ObfuscateBase64Chunks(lang), ObfuscateBase64ChunksMetadata(lang)
        ]
        if lang == 'py':
            single_round_transformations.extend([ObfuscateAPIsPy(lang), ObfuscateAPIsPyMetadata(lang)])
        else:
            raise Exception("Unsupported language: {}".format(lang))
        
        start_time = perf_counter()
        self.num_queries = 0
        self.traces = []
        pkg_idx = 0
        base_name = f"{package.name}-{package.version}"
        if out_log is not None:
            file = open(out_log, 'w')
            logger = file.write
            endl = '\n'
        else:
            logger = print
            endl = ''

        self.best_score = self.model.classify(package)
        if self.debug:
            logger(f"Initial score: {self.best_score:.3f}{endl}")
        self.best_sample = package
        self.traces.append((0, self.best_score, ""))

        for transformation in single_round_transformations:
            pkg_idx += 1
            mutated_pkg = self.best_sample.clone(os.path.join(self.pool_dir, base_name + f"_{pkg_idx}"))
            transformation(mutated_pkg)
            candidates = [(mutated_pkg, type(transformation).__name__)]
            self._evaluate(candidates)
        if self.debug:
            logger(f"Score after single-round transformations: {self.best_score:.3f}{endl}")

        if not self.only_sr:
            for round_idx in range(self.num_rounds):
                candidates = []
                for transformation in multi_round_transformations:
                    pkg_idx += 1
                    mutated_pkg = self.best_sample.clone(os.path.join(self.pool_dir, base_name + f"_{pkg_idx}"))
                    transformation(mutated_pkg)
                    candidates.append((mutated_pkg, type(transformation).__name__))

                self._evaluate(candidates)
                if self.debug:
                    logger(f"Score after round #{round_idx}: {self.best_score:.3f}{endl}")

        run_time = perf_counter() - start_time

        shutil.copytree(self.best_sample.path, package.path, dirs_exist_ok=True)
        for path in glob.glob(os.path.join(self.pool_dir, base_name + "*")):
            shutil.rmtree(path)

        if self.debug:
            logger(f"Reached confidence {self.best_score:.3f}, runtime: {run_time:.4f} s\n{endl}")
        if out_log is not None:
            file.close()

        return self.best_score, self.best_sample, self.num_queries, run_time, self.traces


class OptimizerAdvanced(Optimizer):

    def __init__(self, ml_model, num_rounds, num_candidates, num_transformations_round, only_sr=False, pool_dir=PACKAGES_TEMP_POOL, save_only_best=False, debug=False):
        super().__init__(ml_model, num_rounds, only_sr, pool_dir, save_only_best, debug)
        self.num_transformations_round = num_transformations_round
        self.num_candidates = num_candidates

    def _evaluate(self, candidates):
        """
        Evaluate a set of candidates (the mutated phishing webpages generated from the current best adversarial example),
        and uodate the best adversarial example found so far each time we found a better candidate.

        Arguments:
            candidates: a list of mutated packages (instance of Package class) generated from the current best adversarial example
        """
        for package in candidates:
            score = self.model.classify(package)
            self.num_queries += 1

            if score <= self.best_score:
                self.best_score = score
                self.best_sample = package
                if self.save_only_best:
                    self.traces.append((self.num_queries, self.best_score))
                if self.debug:
                    print("Score after {} queries: {:.3f}".format(self.num_queries, self.best_score))
            else:
                shutil.rmtree(package.path)

            if not self.save_only_best:
                self.traces.append((self.num_queries, score))

    def optimize(self, package: Package):
        """
        Generates an optimized adversarial package

        Arguments:
            package: input package (i.e., an object of Packsge class).

        Returns:
            tuple: a tuple representing the reult of the optimization:
                   * self.best_score: the best score
                   * self.best_sample: the final adversarial package
                   * self.num_queries: the number of used queries
                   * run_time: the run time in seconds
                   * self.traces: the optimization traces
        """
        start_time = perf_counter()
        self.num_queries = 0
        pkg_idx = 0
        base_name = f"{package.name}-{package.version}" 

        self.best_score = self.model.classify(package)
        if self.debug:
            print("Initial score: {:.3f}".format(self.best_score))
        self.best_sample = package
        self.traces.append((0, self.best_score, ""))

        for transformation in OptimizerAdvanced.single_round_transformations:
            pkg_idx += 1
            mutated_pkg = self.best_sample.clone(os.path.join(self.pool_dir, base_name + f"_{pkg_idx}"))
            transformation(mutated_pkg)
            candidates = [mutated_pkg]
            self._evaluate(candidates)
        if self.debug:
            print("Score after single-round transformations: {:.3f}".format(self.best_score))

        for round in range(self.num_rounds):
            candidates = []
            for _ in range(self.num_candidates):
                pkg_idx += 1
                candidate_pkg = self.best_sample.clone(os.path.join(self.pool_dir, base_name + f"_{pkg_idx}"))
                transformations = random.choices(OptimizerAdvanced.multi_round_transformations, k=self.num_transformations_round)
                for transformation in transformations:
                    transformation(candidate_pkg)
                candidates.append(candidate_pkg)

            self._evaluate(candidates)
            if self.debug:
                print("Score after round #{}: {:.3f}".format(round, self.best_score))

        run_time = perf_counter() - start_time

        shutil.copytree(self.best_sample.path, package.path, dirs_exist_ok=True)

        if self.debug:
            print("Reached confidence {}, runtime: {:.4f} s\n".format(self.best_score, run_time))

        return self.best_score, self.best_sample, self.num_queries, run_time, self.traces