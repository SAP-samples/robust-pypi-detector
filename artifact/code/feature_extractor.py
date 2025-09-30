import numpy as np
import os
import pandas as pd
from pygments.lexers import PythonLexer, JavascriptLexer, JsonLexer
from pygments.token import Token
from suspicious_tokens import SUSPICIOUS_TOKENS
import re
from utils import check_base64, get_stats_tokens
import nltk
from pathlib import Path
from utils import Package, gen_language_3, gen_language_4, gen_language_8, gen_language_16, shannon_entropy, get_num_heterogeneous_tokens
import signal
import multiprocessing
import base64


STOPWORDS = set(nltk.corpus.stopwords.words('english'))
TIMEOUT_DEFAULT = 1200  # 20 minutes = 1200 seconds


class FeaturePackage:
    def __init__(self, lang='py') -> None:
        self._lang = lang

    def __call__(self, package):
        self.extract_feature(package)

    def extract_feature(self):
        raise NotImplementedError("This method must be implemented")


class NumWords(FeaturePackage):
    def __call__(self, package):
        return self.extract_feature(package.code_path)

    def extract_feature(self, file_paths):
        num_words = 0
        lexer = PythonLexer() if self._lang == 'py' else JavascriptLexer()

        for file_path in file_paths:
            try:
                with open(file_path, 'r') as file:
                    file_content = file.read()
            except Exception:
                continue

            num_words += len(list(lexer.get_tokens(file_content)))

        return num_words


class NumWordsMetadata(NumWords):
    def __call__(self, package):
        return self.extract_feature(package.metadata_path)


class NumSuspiciousTokens(FeaturePackage):
    def __call__(self, package):
        return self.extract_feature(package.code_path)

    def extract_feature(self, file_paths):
        num_susp_tokens = 0
        lexer = PythonLexer() if self._lang == 'py' else JavascriptLexer()

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
            except Exception:
                continue

            tokens = lexer.get_tokens(file_content)
            for token_type, token_value in tokens:
                if "Token.Literal.String" in str(token_type):
                    if token_value in SUSPICIOUS_TOKENS:
                        num_susp_tokens += 1

        return num_susp_tokens


class NumSuspiciousTokensMetadata(NumSuspiciousTokens):
    def __call__(self, package):
        return self.extract_feature(package.metadata_path)


class NumLines(FeaturePackage):
    def __call__(self, package):
        return self.extract_feature(package.code_path)

    def extract_feature(self, file_paths):
        num_lines = 0

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    num_lines += len(file.readlines())
            except Exception:
                continue

        return num_lines


class NumLinesMetadata(NumLines):
    def __call__(self, package):
        return self.extract_feature(package.metadata_path)


class NumIPs(FeaturePackage):
    def __call__(self, package):
        return self.extract_feature(package.code_path)

    def extract_feature(self, file_paths):
        ip_pattern = r'\b[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b'
        num_ip_address = 0
        lexer = PythonLexer() if self._lang == 'py' else JavascriptLexer()

        for file_path in file_paths:
            try:
                with open(file_path, "r") as file:
                    file_content = file.read()
            except Exception:
                continue

            # num_ip_address += len(re.findall(ip_pattern, file_content))
            tokens = lexer.get_tokens(file_content)
            for token_type, token_value in tokens:
                if "Token.Literal.String" in str(token_type):
                    num_ip_address += len(list(re.finditer(ip_pattern, token_value)))

        return num_ip_address


class NumIPsMetadata(NumIPs):
    def __call__(self, package):
        return self.extract_feature(package.metadata_path)


class NumURLs(FeaturePackage):
    def __call__(self, package):
        return self.extract_feature(package.code_path)

    def extract_feature(self, file_paths):
        # url_pattern = r"""((?:(?:https|http|hxxp|ssh|ftp|sftp|ws|wss|dns|file|git|jni|imap|ldap|ldaps|nfs|smb|smbs|telnet|udp|vnc)?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|org|uk)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|uk|ac)\b/?(?!@)))"""
        url_pattern = r'\b(?:https|http|hxxp):\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)'
        num_urls = 0
        lexer = PythonLexer() if self._lang == 'py' else JavascriptLexer()

        for file_path in file_paths:
            try:
                with open(file_path, "r") as file:
                    file_content = file.read()
            except Exception:
                continue

            # num_urls += len(re.findall(url_pattern, file_content))
            tokens = lexer.get_tokens(file_content)
            for token_type, token_value in tokens:
                if "Token.Literal.String" in str(token_type):
                    num_urls += len(list(re.finditer(url_pattern, token_value)))

        return num_urls


class NumURLsMetadata(NumURLs):
    def __call__(self, package):
        return self.extract_feature(package.code_path)


class NumBase64Chunks(FeaturePackage):
    def __call__(self, package):
        return self.extract_feature(package.code_path)

    def _is_base64(self, string):
        try:
            if isinstance(string, str):
                # If there's any unicode here, an exception will be thrown and the function will return false
                sb_bytes = bytes(string, 'ascii')
            elif isinstance(string, bytes):
                sb_bytes = string
            else:
                raise ValueError("Argument must be string or bytes")
            decoded_string = base64.b64decode(sb_bytes).decode("utf-8")
            decoded_string = ' '.join(decoded_string.split())
            if (decoded_string.isprintable()):
                return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
            else:
                return False
        except Exception:
            return False

    def extract_feature(self, file_paths):
        # b64_pattern = r'(?:[A-Za-z0-9+\/]{4})*(?:[A-Za-z0-9+\/]{2}==|[A-Za-z0-9+\/]{3}=|[A-Za-z0-9+\/]{4})'
        # The following patterns are more restrictive than the previous one but can be used to avoid false positives
        base64_patterns = [
            # r'(?:[A-Za-z0-9+\/]{4})*(?:[A-Za-z0-9+\/]{2}==|[A-Za-z0-9+\/]{3}=|[A-Za-z0-9+\/]{4})',
            (r"b64decode\s*\(\s*b?('|\")((?:[A-Za-z0-9+\/]{4})*(?:[A-Za-z0-9+\/]{2}==|[A-Za-z0-9+\/]{3}=|[A-Za-z0-9+\/]{4}))\1\s*\)", 2),
            (r'powershell.+?-EncodedCommand\s+((?:[A-Za-z0-9+\/]{4})*(?:[A-Za-z0-9+\/]{2}==|[A-Za-z0-9+\/]{3}=|[A-Za-z0-9+\/]{4}))', 1)
        ]

        num_b64_chunks = 0

        for file_path in file_paths:
            try:
                with open(file_path, "r") as file:
                    file_content = file.read()
            except Exception:
                continue

            num_b64_chunks += len([match.group(group_idx) 
                                   for base64_pattern, group_idx in base64_patterns
                                   for match in re.finditer(base64_pattern, file_content) 
                                   if check_base64(match.group(group_idx))])

            # lexer = PythonLexer() if self._lang == 'py' else JavascriptLexer()
            # tokens = lexer.get_tokens(file_content)
            # base64_pattern = r'[a-zA-Z0-9=/\+]*'
            # for token_type, token_value in tokens:
            #     if "Token.Literal.String" in str(token_type):
            #         if re.match(base64_pattern, token_value) and self._is_base64(token_value):
            #             num_b64_chunks += 1

        return num_b64_chunks


class NumBase64ChunksMetadata(NumBase64Chunks):
    def __call__(self, package):
        return self.extract_feature(package.metadata_path)


class GetStatsToken(FeaturePackage):
    def __call__(self, package, target_token):
        return self.extract_feature(package.code_path, target_token)

    def extract_feature(self, file_paths, target_token):
        num_tokens_files = []
        lexer = PythonLexer() if self._lang == 'py' else JavascriptLexer()

        for file_path in file_paths:
            num_tokens = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
            except Exception:
                continue

            tokens = lexer.get_tokens(file_content)
            for token_type, token_value in tokens:
                # if (target_token == '[' and isinstance(token_type, Token.Punctuation)) or \
                #         (target_token in {'+', '='} and isinstance(token_type, Token.Operator)):
                if token_value == target_token:
                    num_tokens += 1
            num_tokens_files.append(num_tokens)

        mean, max, std, q3 = get_stats_tokens(num_tokens_files)

        return mean, max, std, q3


class GetStatsBrackets(GetStatsToken):
    def __call__(self, package):
        return self.extract_feature(package.code_path, target_token='[')


class GetStatsBracketsMetadata(GetStatsToken):
    def __call__(self, package):
        return self.extract_feature(package.metadata_path, target_token='[')


class GetStatsPlus(GetStatsToken):
    def __call__(self, package):
        return self.extract_feature(package.code_path, target_token='+')


class GetStatsPlusMetadata(GetStatsToken):
    def __call__(self, package):
        return self.extract_feature(package.metadata_path, target_token='+')


class GetStatsEqual(GetStatsToken):
    def __call__(self, package):
        return self.extract_feature(package.code_path, target_token='=')


class GetStatsEqualMetadata(GetStatsToken):
    def __call__(self, package):
        return self.extract_feature(package.metadata_path, target_token='=')


class GetStatsGeneralizationLanguage(FeaturePackage):
    def __call__(self, package, token_cat, gen_language_type, base_se):
        return self.extract_feature(package.code_path, token_cat, gen_language_type, base_se)

    def extract_feature(self, file_paths, tokens_cat='id', gen_language_type=4, base_se=4):
        assert tokens_cat in ['id', 'string']
        assert gen_language_type in [3, 4, 8, 16]
        if gen_language_type == 3:
            gen_language = gen_language_3
        elif gen_language_type == 4:
            gen_language = gen_language_4
        elif gen_language_type == 8:
            gen_language = gen_language_8
        elif gen_language_type == 16:
            gen_language = gen_language_16

        lexer = PythonLexer() if self._lang == 'py' else JavascriptLexer()

        selected_tokens = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
            except Exception:
                continue

            tokens = lexer.get_tokens(file_content)
            for token_type, token_value in tokens:
                if tokens_cat == 'id' and str(token_type) == "Token.Name":
                    token = token_value.replace("'", "").replace('"', "")
                    selected_tokens.append(token)
                elif tokens_cat == 'string' and "Token.Literal.String" in str(token_type):
                    token = token_value.replace("'", "").replace('"', "")
                    if token not in STOPWORDS:
                        selected_tokens.append(token)

        # apply the generalization language
        generalization_tokens = []
        for h in range(0, len(selected_tokens)):
            gen = gen_language(selected_tokens[h])
            generalization_tokens.append(gen)
        num_heterogeneous_tokens = get_num_heterogeneous_tokens(generalization_tokens, symbols=["u", "d", "l", "s"])
        # apply shannon entropy
        shannon_entropy_values = []
        for word in generalization_tokens:
            entropy = shannon_entropy(word, base_se)
            shannon_entropy_values.append(entropy)
        num_homogeneous_tokens = len(list(filter(lambda x: abs(x) == 0, shannon_entropy_values)))
        mean_se, max_se, std_se, q3_se = get_stats_tokens(shannon_entropy_values)

        return mean_se, std_se, max_se, q3_se, num_homogeneous_tokens, num_heterogeneous_tokens


class GetStatsGLStrings(GetStatsGeneralizationLanguage):
    def __call__(self, package):
        return self.extract_feature(package.code_path, tokens_cat='string')


class GetStatsGLStringsMetadata(GetStatsGeneralizationLanguage):
    def __call__(self, package):
        return self.extract_feature(package.metadata_path, tokens_cat='string')


class GetStatsGLIdentifiers(GetStatsGeneralizationLanguage):
    def __call__(self, package):
        return self.extract_feature(package.code_path, tokens_cat='id')


class GetStatsGLIdentifiersMetadata(GetStatsGeneralizationLanguage):
    def __call__(self, package):
        return self.extract_feature(package.metadata_path, tokens_cat='id')


class GetNumFilesType(FeaturePackage):
    def __init__(self, lang='py') -> None:
        super().__init__(lang)

        self.file_ext = ['bat', 'bz2', 'c', 'cert', 'conf', 'cpp', 'crt', 'css', 'csv', 'deb', 'erb', 'gemspec', 'gif', 'gz', 'h', 'html', 'ico', 'ini', 'jar', 'java', 'jpg', 'js', 'json',
                         'key', 'm4v', 'markdown', 'md', 'pdf', 'pem', 'png', 'ps', 'py', 'rb', 'rpm', 'rst', 'sh', 'svg', 'toml', 'ttf', 'txt', 'xml', 'yaml', 'yml', 'eot', 'exe', 'jpeg',
                         'properties', 'sql', 'swf', 'tar', 'woff', 'woff2', 'aac', 'bmp', 'cfg', 'dcm', 'dll', 'doc', 'flac', 'flv', 'ipynb', 'm4a', 'mid', 'mkv', 'mp3', 'mp4', 'mpg',
                         'ogg', 'otf', 'pickle', 'pkl', 'psd', 'pxd', 'pxi', 'pyc', 'pyx', 'r', 'rtf', 'so', 'sqlite', 'tif', 'tp', 'wav', 'webp', 'whl', 'xcf', 'xz', 'zip', 'mov', 'wasm', 'webm']


    def __call__(self, package):
        return self.extract_feature(package.path)

    def extract_feature(self, base_path):
        extension_counts = {ext: 0 for ext in self.file_ext}

        for ext in self.file_ext:
            for file_path in Path(base_path).rglob(f'*.{ext}'):
                extension_counts[ext] += 1

        return list(extension_counts.values())


class PresenceInstallScript(FeaturePackage):
    def __call__(self, package):
        return self.extract_feature(package.metadata_path)

    def extract_feature(self, metadata_path):
        # install_script_regex = r"cmdclass\s*=\s*{(('|")[a-zA-Z_]\w*\2\s*:\s*[a-zA-Z_]\w*\s*,?)*\s*('|")install\3\s*:\s*[a-zA-Z_]\w*,?\s*(('|")[a-zA-Z_]\w*\5\s*:\s*[a-zA-Z_]\w*\s*,?)*}"

        target_filename = 'setup.py' if self._lang == 'py' else 'package.json'
        lexer = PythonLexer() if self._lang == 'py' else JsonLexer()
        if isinstance(metadata_path, list) and len(metadata_path) == 1 and metadata_path[0].endswith(target_filename):
            try:
                with open(metadata_path[0], 'r', encoding='utf-8') as file:
                    file_content = file.read()
            except Exception:
                return 0.0

            # return 1.0 if re.search(install_script_regex, file_content) else 0.0

            tokens = lexer.get_tokens(file_content)
            for token_type, token_value in tokens:
                if "Token.Literal.String" in str(token_type):
                    if self._lang == 'py':
                        if re.match("install", token_value):
                            return 1.0
                    else:
                        # NOTE: There could be other commands to run scripts during life-cycle installation such as:
                        # "preprepare", "prepare", "postprepare", "prepublish", "publish", "postpublish", ...
                        # See also: https://docs.npmjs.com/cli/v10/using-npm/scripts
                        if re.match("preinstall", token_value) or re.match("install", token_value) or re.match("postinstall", token_value):
                            return 1.0
        return 0.0


def extract_features_package(package: Package):
    language = 'py' if package.type == 'pypi' else 'js'

    num_words_source = NumWords(language)(package)
    num_lines_source = NumLines(language)(package)
    plus_ratio_mean, plus_ratio_max, plus_ratio_std, plus_ratio_q3 = GetStatsPlus(language)(package)
    eq_ratio_mean, eq_ratio_max, eq_ratio_std, eq_ratio_q3 = GetStatsEqual(language)(package)
    bracket_ratio_mean, bracket_ratio_max, bracket_ratio_std, bracket_ratio_q3 = GetStatsBrackets(language)(package)
    num_base64_chunks_source = NumBase64Chunks(language)(package)
    num_ips_source = NumIPs(language)(package)
    num_susp_tokens_source = NumSuspiciousTokens(language)(package)
    num_urls_source = NumURLs(language)(package)

    num_words_metadata = NumWordsMetadata(language)(package)
    num_lines_source = NumLinesMetadata(language)(package)
    num_base64_chunks_metadata = NumBase64ChunksMetadata(language)(package)
    num_ips_metadata = NumIPsMetadata(language)(package)
    num_susp_tokens_metadata = NumSuspiciousTokensMetadata(language)(package)
    num_urls_metadata = NumURLsMetadata(language)(package)

    num_files_types = GetNumFilesType(language)(package)
    presence_install_script = PresenceInstallScript(language)(package)
    mean_se_id_source, std_se_id_source, max_se_id_source, q3_se_id_source, num_homogeneous_id_source, num_heterogeneous_id_source = GetStatsGLIdentifiers(language)(package)
    mean_se_id_metadata, std_se_id_metadata, max_se_id_metadata, q3_se_id_metadata, num_homogeneous_id_metadata, num_heterogeneous_id_metadata = GetStatsGLIdentifiersMetadata(language)(package)
    mean_se_string_source, std_se_string_source, max_se_string_source, q3_se_string_source, num_homogeneous_string_source, num_heterogeneous_string_source = GetStatsGLStrings(language)(package)
    mean_se_string_metadata, std_se_string_metadata, max_se_string_metadata, q3_se_string_metadata, num_homogeneous_string_metadata, num_heterogeneous_string_metadata = GetStatsGLStringsMetadata(language)(package)

    package_features = [num_words_source, num_lines_source, plus_ratio_mean, plus_ratio_max, plus_ratio_std, plus_ratio_q3, eq_ratio_mean, eq_ratio_max, eq_ratio_std, eq_ratio_q3, bracket_ratio_mean, bracket_ratio_max, bracket_ratio_std, bracket_ratio_q3,
        num_base64_chunks_source, num_ips_source, num_susp_tokens_source, num_words_metadata, num_lines_source, num_base64_chunks_metadata, num_ips_metadata, num_susp_tokens_metadata] + num_files_types + \
        [presence_install_script, mean_se_id_source, std_se_id_source, max_se_id_source, q3_se_id_source, mean_se_string_source, std_se_string_source, max_se_string_source, q3_se_string_source,
            num_homogeneous_id_source, num_homogeneous_string_source, num_heterogeneous_id_source, num_heterogeneous_string_source, num_urls_source,
            mean_se_id_metadata, std_se_id_metadata, max_se_id_metadata, q3_se_id_metadata, mean_se_string_metadata, std_se_string_metadata, max_se_string_metadata, q3_se_string_metadata,
            num_homogeneous_id_metadata, num_homogeneous_string_metadata, num_heterogeneous_string_metadata, num_urls_metadata, num_heterogeneous_id_metadata]

    return package_features


def extract_features_sample(package, label):
    def _signal_handler(signum, frame):
        raise TimeoutError()

    # Timeout setup
    signal.signal(signal.SIGALRM, _signal_handler)
    signal.alarm(TIMEOUT_DEFAULT)
    try:
        features = [f'{package.name}-{package.version}'] + extract_features_package(package)
    except TimeoutError:
        print(f"[WARN] Timeout error for {package.name}-{package.version}")
        features = []
    else:
        features.append(label)
        signal.alarm(0)
        # print(f"Features extracted for {package.name}-{package.version}")
    return features


class FeatureExtractor:
    def __init__(self) -> None:
        nltk.download('stopwords', quiet=True)

    def extract_features(self, packages, label, n_jobs=None):

        assert n_jobs is None or (isinstance(n_jobs, int) and n_jobs > 0)
        if n_jobs is None:
            n_jobs = len(os.sched_getaffinity(0)) - 1
        # print(f"[INFO] Extracting features using {n_jobs} parallel jobs")

        columns = ["Package Name", "Number of Words in source code", "Number of lines in source code", "plus ratio mean", "plus ratio max", "plus ratio std", "plus ratio q3", "eq ratio mean", "eq ratio max", "eq ratio std", "eq ratio q3",
                   "bracket ratio mean", "bracket ratio max", "bracket ratio std", "bracket ratio q3", "Number of base64 chunks in source code", "Number of IP address in source code", "Number of suspicious token in source code",
                   "Number of Words in metadata", "Number of lines in metadata", "Number of base64 chunks in metadata", "Number of IP address in metadata", "Number of suspicious token in metadata",
                   ".bat", ".bz2", ".c", ".cert", ".conf", ".cpp", ".crt", ".css", ".csv", ".deb", ".erb", ".gemspec", ".gif", ".gz", ".h", ".html", ".ico", ".ini", ".jar", ".java", ".jpg", ".js", ".json", ".key", ".m4v", ".markdown", ".md", ".pdf", ".pem", ".png", ".ps", ".py",
                   ".rb", ".rpm", ".rst", ".sh", ".svg", ".toml", ".ttf", ".txt", ".xml", ".yaml", ".yml", ".eot", ".exe", ".jpeg", ".properties", ".sql", ".swf", ".tar", ".woff", ".woff2", ".aac", ".bmp", ".cfg", ".dcm", ".dll", ".doc", ".flac", ".flv", ".ipynb", ".m4a", ".mid",
                   ".mkv", ".mp3", ".mp4", ".mpg", ".ogg", ".otf", ".pickle", ".pkl", ".psd", ".pxd", ".pxi", ".pyc", ".pyx", ".r", ".rtf", ".so", ".sqlite", ".tif", ".tp", ".wav", ".webp", ".whl", ".xcf", ".xz", ".zip", ".mov", ".wasm", ".webm",
                   "presence of installation script", "shannon mean ID source code", "shannon std ID source code", "shannon max ID source code", "shannon q3 ID source code", "shannon mean string source code", "shannon std string source code", "shannon max string source code", "shannon q3 string source code",
                   "homogeneous identifiers in source code", "homogeneous strings in source code", "heterogeneous identifiers in source code", "heterogeneous strings in source code", "URLs in source code", "shannon mean ID metadata", "shannon std ID metadata", "shannon max ID metadata", "shannon q3 ID metadata",
                   "shannon mean string metadata", "shannon std string metadata", "shannon max string metadata", "shannon q3 string metadata", "homogeneous identifiers in metadata", "homogeneous strings in metadata", "heterogeneous strings in metadata", "URLs in metadata", "heterogeneous identifiers in metadata", "label"]

        assert isinstance(label, int) or (isinstance(label, list) and len(label) == len(packages))
        labels = [label] * len(packages) if isinstance(label, int) else label

        # results = Parallel(n_jobs=n_jobs, timeout=TIMEOUT_DEFAULT)(delayed(extract_features_sample)(package, y) for package, y in zip(packages, labels))
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = pool.starmap(extract_features_sample, zip(packages, labels))
        results = [res for res in results if res]

        df_packages = pd.DataFrame(data=results, columns=columns)
        assert not df_packages.isnull().values.any()

        return df_packages