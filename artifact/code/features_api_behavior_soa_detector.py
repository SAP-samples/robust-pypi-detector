# from fcg import get_fcg
import os
import json
# from run_training import get_packages
import re
import pandas as pd


SENSITIVE_API_NETWORK = {
    "requests": ["get", "post", "put", "patch", "delete", "request", "Session"],
    "socket": ["socket", "close", "create_connection", "create_server", "dup",
                #"getaddrinfo", "getfqdn", "gethostbyname", "gethostname", "getnameinfo","getprotobyname", "getservbyname"
                ],
    "socket.socket": ["bind", "listen", "accept", "connect", "connect_ex", "close", "detach", "dup", "shutdown"
                      "send", "sendall", "sendto", "sendfile", "sendmsg",
                      "recv", "recvfrom", "recv_into", "recvfrom_into", "recvmsg",
                      #"fileno", "getpeername", "getsockname", "getsockopt", "setsockopt", "setblocking", "settimeout"
                      ],
    "webhook": ["send"],
    "Webhook": ["from_url"],  # Webhook from discord
    "aiohttp": ["request"],
    "aiohttp.ClientSession": ["get", "post", "put", "patch", "delete", "ws_connect"],
    "http.client.HTTPConnection": ["request", "getresponse", "putrequest", "connect", "close"],
    "http.client.HTTPSConnection": ["request", "getresponse", "putrequest", "connect", "close"],
    "urllib.request": ["urlopen", "urlretrieve", "Request"],  # urllib.request
    "urllib3": ["request"],  # urllib3
    "urllib3.connection.HTTPConnection": ["connect", "request", "request_chunked", "getresponse", "close"],
    "urllib3.connection.HTTPSConnection": ["connect", "request", "request_chunked", "getresponse", "close"],
    # "webdriver": ["Chrome", "get"]
}

SENSITIVE_API_FILESYSTEM = {
    # add security-sensitive api for filesystem operations and host info
    "os": ["open", "remove", "rename", "replace", "truncate", "stat", "lstat", "fstat", "chown", "chmod", "lchmod", "lchown", "link", "symlink", "readlink", "realpath", "unlink", "rmdir", "mkdir", "makedirs", "removedirs", "walk", "listdir", "chdir", "fchdir", "access", "startfile", "mkfifo", "mknod", "pathconf", "fpathconf", "statvfs", "fstatvfs", "fsync", "fdatasync", "sync", "fsync_range"],
    "shutil": ["copyfile", "copyfileobj", "copy", "copy2", "copytree", "rmtree", "move"],
    "io.BufferedReader": ["read", "read1", "readinto", "readinto1"],
    "io.BufferedWriter": ["write"],
    "open": [],  # may generate false positives
    # "builtins": ["open"],  # may generate false positives
    "read": [],
    "write": [],
    # "": ["read", "write"]
}

SENSITIVE_API_HOSTINFO = {
    "getpass": ["getpass", "getuser"],
    "socket": ["gethostname", "fileno", "getpeername", "gethostbyname"],
    "os": ["getcwd", "getlogin", "getpid", "getppid", "getuid", "geteuid", "getgid", "getegid", "getgroups", "getenv", "environ"],
    "platform": ["node", "system", "release", "version", "machine", "processor", "architecture", "platform", "uname", "linux_distribution", "mac_ver", "win32_ver"],
    "winreg": ["CreateKey", "CreateKeyEx", "ConnectRegistry", "DeleteKey", "DeleteKeyEx", "DeleteValue", "EnumKey", "EnumValue", "LoadKey", "OpenKey", "OpenKeyEx", "QueryValue", "QueryValueEx", "SaveKey", "SetValue", "SetValueEx", "QueryInfoKey"],
}


SENSITIVE_API_CMD_EXEC = {
    "subprocess": ["getoutput", "call", "check_output", "run", "Popen", "check_call"],
    "pty": ["fork", "openpty", "spawn"],
    "os": ["popen", "system", "posix_spawn", "posix_spawnp", "getenv", "chmod", "dup2", "startfile"] + [f"exec{mode}{submode}" for mode in ['l', 'v'] for submode in ['', 'e', 'p', 'pe']] + [f"spawn{mode}{submode}" for mode in ['l', 'v'] for submode in ['', 'e', 'p', 'pe']],
}

SENSITIVE_API_CODE_EXEC = {
    "exec": [],
    "eval": []
}

SENSITIVE_API_ENCODING = {
    "base64": ["b64encode", "b64decode", "urlsafe_b64encode", "urlsafe_b64decode", "standard_b64encode", "standard_b64decode", "b32encode", "b32decode", "b16encode", "b16decode", "b85encode", "b85decode", "encode", "decode"],
    "hashlib": ["md5", "sha1", "sha224", "sha256", "sha384", "sha512"],
    "bytearray": ["fromhex", "hex"],
    "zlib": ["compress", "decompress"],
    "gzip": ["compress", "decompress"],
    "lzma": ["compress", "decompress"],
    "marshal": ["load", "loads"],
    "__pyarmor__": [],
    # "encode": [],
    # "decode": []
}


# MALICIOUS BEHAVIORS
REMOTE_CONTROL = [
    ["NETWORK", "NETWORK", "ENCODING", "CMDEXEC"],
    ["NETWORK", "NETWORK", "CMDEXEC"],
    ["NETWORK", "CMDEXEC"]
]

# REMOTE_CONTROL_RE = r"NETWORK_(NETWORK_)?(ENCODING_)?CMDEXEC"
# regex with gaps
REMOTE_CONTROL_RE = r"NETWORK_(NETWORK_)?\w*(ENCODING_)?\w*CMDEXEC"

INFORMATION_STEALING = [
    ["FILESYSTEM", "NETWORK"],
    ["FILESYSTEM", "HOSTINFO", "NETWORK"],
    ["FILESYSTEM", "ENCODING", "NETWORK"],
    ["HOSTINFO", "NETWORK"],
    ["HOSTINFO", "ENCODING", "NETWORK"],
    ["HOSTINFO", "FILESYSTEM", "NETWORK"]
]

# INFORMATION_STEALING_RE = r"FILESYSTEM_(HOSTINFO_|ENCODING_)?NETWORK|HOSTINFO_(FILESYSTEM_|ENCODING_)?NETWORK"
# regex with gaps
INFORMATION_STEALING_RE = r"FILESYSTEM_(HOSTINFO_|ENCODING_)?\w*NETWORK|HOSTINFO_(FILESYSTEM_|ENCODING_)?\w*NETWORK"

CODE_EXECUTION = [
    ["NETWORK", "ENCODING", "CEXEC"],
    ["NETWORK", "CEXEC"],
    ["ENCODING", "CEXEC"],
    ["CEXEC"]
]

# CODE_EXECUTION_RE = r"(NETWORK_)?(ENCODING_)?CEXEC"
# regex with gaps
CODE_EXECUTION_RE = r"NETWORK_(ENCODING_)?\w*CEXEC|ENCODING_\w*CEXEC|CEXEC"

COMMAND_EXECUTION = [
    # ["CMDEXEC"],
    ["CMDEXEC", "NETWORK"],
    ["CMDEXEC", "ENCODING"],
    ["CMDEXEC", "ENCODING", "NETWORK"],
    ["ENCODING", "CMDEXEC"],
    ["ENCODING", "CMDEXEC", "NETWORK"],
    ["ENCODING", "CMDEXEC", "ENCODING"],
    ["ENCODING", "CMDEXEC", "ENCODING", "NETWORK"]
]

# COMMAND_EXECUTION_RE = r"(ENCODING_)?CMDEXEC(_ENCODING(_NETWORK)?|_NETWORK)?"
COMMAND_EXECUTION_RE = r"CMDEXEC_(ENCODING_)?\w*NETWORK|ENCODING_\w*CMDEXEC_(ENCODING_)?\w*NETWORK|CMDEXEC_\w*ENCODING|ENCODING_\w*CMDEXEC_\w*ENCODING|ENCODING_\w*CMDEXEC"

UNAUTH_FILE_OPERATION = [
    ["NETWORK", "FILESYSTEM", "CMDEXEC", "FILESYSTEM"],
    ["NETWORK", "FILESYSTEM", "CMDEXEC"],
    ["FILESYSTEM", "CMDEXEC", "FILESYSTEM"],
    ["FILESYSTEM", "CMDEXEC"],
]

# UNAUTH_FILE_OPERATION_RE = r"(NETWORK_)?FILESYSTEM_CMDEXEC(_FILESYSTEM)?"
# regex with gaps
UNAUTH_FILE_OPERATION_RE = r"NETWORK_\w*FILESYSTEM_\w*CMDEXEC_\w*FILESYSTEM|NETWORK_\w*FILESYSTEM_\w*CMDEXEC|FILESYSTEM_\w*CMDEXEC_\w*FILESYSTEM|FILESYSTEM_\w*CMDEXEC"

BEHAVIORS_RE = {
    'REMOTE_CONTROL': REMOTE_CONTROL_RE,
    'INFORMATION_STEALING': INFORMATION_STEALING_RE,
    'CODE_EXECUTION': CODE_EXECUTION_RE,
    'COMMAND_EXECUTION': COMMAND_EXECUTION_RE,
    'UNAUTH_FILE_OPERATION': UNAUTH_FILE_OPERATION_RE
}


COMMAND_EXECUTION_EXTENDED = COMMAND_EXECUTION + REMOTE_CONTROL + UNAUTH_FILE_OPERATION

BEHAVIORS = {
    'REMOTE_CONTROL': REMOTE_CONTROL,
    'INFORMATION_STEALING': INFORMATION_STEALING,
    'CODE_EXECUTION': CODE_EXECUTION,
    'COMMAND_EXECUTION': COMMAND_EXECUTION,
    'UNAUTH_FILE_OPERATION': UNAUTH_FILE_OPERATION
}

BEHAVIORS_NEW = {
    'INFO_STEALING': INFORMATION_STEALING,
    'CODE_EXECUTION': CODE_EXECUTION,
    'COMMAND_EXECUTION': COMMAND_EXECUTION_EXTENDED
}


def api2category():
    apis_category = {}

    def assign_category(api_dict, category):
        for module, apis in api_dict.items():
            if apis:
                for api in apis:
                    apis_category[f'{module}.{api}'] = category
            else:
                apis_category[module] = category

    assign_category(SENSITIVE_API_NETWORK, "NETWORK")
    assign_category(SENSITIVE_API_FILESYSTEM, "FILESYSTEM")
    assign_category(SENSITIVE_API_HOSTINFO, "HOSTINFO")
    assign_category(SENSITIVE_API_CMD_EXEC, "CMDEXEC")
    assign_category(SENSITIVE_API_CODE_EXEC, "CEXEC")
    assign_category(SENSITIVE_API_ENCODING, "ENCODING")

    return apis_category


def extract_behaviors_package(package, apis_category):

    def find_category(api):
        for api_pattern, category in apis_category.items():
            if re.search(api_pattern, api):
                return category
        return None

    # create a OR regex pattern including all the APIs of each category
    all_sensitive_apis_re = re.compile("|".join(apis_category.keys()))

    package_behaviors_count = {behavior_name: 0 for behavior_name in BEHAVIORS_RE.keys()}

    package_files = package.code_path + package.metadata_path

    for file in package_files:
        try:
            with open(file, "r") as f:
                content = f.read()
        except Exception as e:
            continue

        # find the list of all the sensistive API calls in the file 
        api_sequence = all_sensitive_apis_re.findall(content)

        # convert each API to a list of categories
        api_sequence_categories = []
        for api in api_sequence:
            category = find_category(api)
            if category:
                api_sequence_categories.append(category)
    
        if api_sequence_categories:
            for behavior_name, behavior_pattern in BEHAVIORS_RE.items():
                # print(f"Checking behavior {behavior_name}")
                api_sequence_categories_str = "_".join(api_sequence_categories)
                matches = list(re.finditer(behavior_pattern, api_sequence_categories_str))
                package_behaviors_count[behavior_name] += len(matches)

    return package_behaviors_count


def get_api_counting(package, api_category, by_category=False):
    package_files = package.code_path + package.metadata_path
    
    if by_category:
        api_counting = {
            "NETWORK": 0,
            "FILESYSTEM": 0,
            "HOSTINFO": 0,
            "CMDEXEC": 0,
            "CEXEC": 0,
            "ENCODING": 0
        }
    else:
        api_list = list(api_category.keys())
        api_counting = {api: 0 for api in api_list}

    for file in package_files:
        try:
            with open(file, "r") as f:
                content = f.read()
        except Exception as e:
            continue

        for api_pattern, category in api_category.items():
            module, api = api_pattern.rsplit(".", 1) if "." in api_pattern else (api_pattern, None)
            if module == "socket.socket" or module == "http.client.HTTPConnection" or module == "http.client.HTTPSConnection" or \
                    module == "urllib3.connection.HTTPConnection" or module == "urllib3.connection.HTTPSConnection" or module == "urllib.request" or \
                    module == "io.BufferedReader" or module == "io.BufferedWriter":
                if re.search(rf"{module}", content) and re.search(rf"{api}\(", content):
                    if by_category:
                        api_counting[category] += len(list(re.finditer(rf"{api}\(", content)))
                    else:
                        api_counting[api_pattern] += len(list(re.findall(rf"{api}\(", content)))
            else:
                if re.search(rf"{api_pattern}\(", content):
                    if by_category:
                        api_counting[category] += len(list(re.finditer(rf"{api_pattern}\(", content)))
                    else:
                        api_counting[api_pattern] += len(list(re.finditer(rf"{api_pattern}\(", content)))
                elif api is not None and re.search(rf"from\s*{module}\s*import.*{api}", content) and re.search(f"{api}\(", content):
                    if by_category:
                        api_counting[category] += len(list(re.finditer(rf"{api}\(", content)))
                    else:
                        api_counting[api_pattern] += len(list(re.finditer(rf"{api}\(", content)))

    return api_counting


def get_api_behavior_features(package):
    apis_category = api2category()
    # fcg, modules_calls_list = get_fcg(package)
    
    # API COUNTING
    api_counting = get_api_counting(package, apis_category, by_category=False)

    # BEHAVIORS COUNTING
    pkg_behaviors = extract_behaviors_package(package, apis_category)

    return list(api_counting.values()), list(pkg_behaviors.values())
