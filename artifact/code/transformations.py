import os
import re
import random
import base64
from utils import random_string, check_base64, create_new_file_py
import glob
import codecs
from pygments.lexers import PythonLexer, JavascriptLexer
from pygments import lex
from pygments.token import Token


# Refs:
# - https://orca.security/resources/blog/understand-shell-commands-detect-malicious-behavior/
# - https://redcanary.com/threat-detection-report/techniques/windows-command-shell/
# - https://www.elastic.co/guide/en/security/current/suspicious-windows-powershell-arguments.html
# - https://github.com/security-cheatsheet/cmd-command-cheat-sheet
SUSPCIOUS_TOKENS_CMDS = [
    # Unix/Linux
    r"\bsh\b",
    r"\bbash\b",
    r"\bcurl\b",
    r"\bwget\b",
    r"\bftp\b",
    r"\btftp\b",
    r"\bperl\b",
    r"\bpython\d{0,1}\b",
    r"\bwhoami\b",
    r"\buname\b",
    r"\bhostname\b",
    r"\bfind\b",
    r"\bcat\b",
    r"\bgrep\b",
    r"\bawk\b",
    r"\bsudo\b",
    r"\bsystemctl\b",
    r"\bchmod\b",
    r"\bchown\b",
    r"\bchgrp\b",
    r"\bkill\b",
    r"\bkillall\b",
    r"\bping\b",
    r"\bnc\b",
    r"\bnetcat\b",
    r"\bssh-\w+\b",
    r"\beval\b",
    r"\binstall\b"
    r"\btar\b",
    r"\bunzip\b",
    # Windows specific
    r"\bpowershell\b",
    r"\bstart\b",
    r"\bexit\b",
    r"\bschtasks\b",
    r"\bspawncmd\b",
    r"-exec\b",
    r"\bcmd\b",
    r"\bbypass\b",
    r"Base64String",
    r"[*Convert]",
    r".Compression.",
    r"-join($",
    r"\bMemoryStream\b",
    r"\bWriteAllBytes\b",
    r"-enc\b",
    r"-ec\b",
    r"\b/e\b",
    r"\b/enc\b",
    r"\b/ec\b",
    r"\bWebClient\b",
    r"\bDownloadFile\b",
    r"\bDownloadString\b",
    r"\bpowercat\b",
    r"$host.UI.PromptForCredential",
    r"\bWhoami\b",
    r"\bTakeown\b",
    r"\bStart\b",
    r"\bSetx\b",
    r"\bReg\b",
    r"\bRegini\b",
    r"-WindowStyle Hidden\b",
    r"-EncodedCommand\b"
    r"\btaskkill\b"
]


class Transformation:
    def __init__(self, lang='py'):
        assert lang in ['py', 'js']
        self._lang = lang

    def __call__(self, package):
        raise NotImplementedError("The transformation functionality must me defined by overriding __call__()")


def obfuscate_string_py(string, mode, **kwargs):
    chunk_size = kwargs.get('chunk_size', 2)
    assert isinstance(chunk_size, int) and chunk_size >= 1
    assert mode in ['bytearray', 'base64', 'base32', 'base16', 'rot13', 'hex', 'split_inline']

    if mode == "base64":
        encoded = base64.b64encode(string.encode()).decode()
        payload = "' + __import__('base64').b64decode('{}').decode() + '".format(encoded)
    elif mode == "base32":
        encoded = base64.b32encode(string.encode()).decode()
        payload = "' + __import__('base64').b32decode('{}').decode() + '".format(encoded)
    elif mode == "base16":
        encoded = base64.b16encode(string.encode()).decode()
        payload = "' + __import__('base64').b16decode('{}').decode() + '".format(encoded)
    elif mode == 'hex':
        encoded = string.encode().hex()
        payload = "' + bytes.fromhex('{}').decode() + '".format(encoded)
    elif mode == 'bytearray':
        encoded = bytes([ord(s) for s in string])
        payload = "' + bytes({}).decode() + '".format([ord(c) for c in string])
    elif mode == 'rot13':
        encoded = codecs.encode(string, 'rot_13')
        payload = "' + __import__('codecs').decode({}, 'rot_13') + '".format(encoded)
    elif mode == 'split_inline':
        chunk_size = 2
        chunks = ["'{}'".format(string[i:i + chunk_size]) for i in range(0, len(string), chunk_size)]
        payload = "' + " + " + ".join(chunks) + " + '"

    return payload


def obfuscate_string_js(string, mode, **kwargs):
    chunk_size = kwargs.get('chunk_size', 2)
    assert isinstance(chunk_size, int) and chunk_size >= 1
    assert mode in ['bytearray', 'base64', 'hex', 'split_inline']

    if mode == "base64":
        encoded = base64.b64encode(string.encode()).decode()
        payload = "' + atob('{}') + '".format(encoded)
    elif mode == 'hex':
        encoded = string.encode().hex()
        payload = "' + Buffer.from('{}', 'hex').toString() + '".format(encoded)
    elif mode == 'bytearray':
        encoded = bytes([ord(s) for s in string])
        payload = "' + String.fromCharCode({}) + '".format(",".join([str(ord(c)) for c in string]))
    elif mode == 'split_inline':
        chunk_size = 2
        chunks = ["'{}'".format(string[i:i + chunk_size]) for i in range(0, len(string), chunk_size)]
        payload = "' + " + " + ".join(chunks) + " + '"

    return payload


class ObfuscateIPs(Transformation):
    def __call__(self, package):
        self._obfuscate(package.code_path)

    def _check_ip(self, ip):
        ip_digits = [int(digit) for digit in ip.split('.')]
        return len(ip_digits) == 4 and all([0 <= digit <= 255 for digit in ip_digits])

    def _obf_rand_base_pad(self, ip):
        # Code adapted from https://github.com/vysecurity/IPFuscator
        hexparts = []
        octparts = []
        randbase = []

        parts = ip.split('.')
        decimal_ip = int(parts[0]) * 16777216 + int(parts[1]) * 65536 + int(parts[2]) * 256 + int(parts[3])
        octal_ip = oct(decimal_ip)[2:]
        hex_ip = hex(decimal_ip)

        for i in parts:
            hexparts.append(hex(int(i)))
            octparts.append("0" + oct(int(i))[2:])

        count = 0
        while count < 5:
            randbaseval = ""
            for i in range(0,4):
                val = random.randint(0,2)
                if val == 0:
                    # dec
                    randbaseval += parts[i] + '.'
                elif val == 1:
                    # hex
                    randbaseval += hexparts[i].replace('0x', '0x' + '0' * random.randint(1,30)) + '.'
                else:
                    randbaseval += '0' * random.randint(1,30) + octparts[i] + '.'
                    # oct
            randbase.append(randbaseval[:-1])
            # print("#{}:\t{}".format(count+1, randbase[count]))
            count += 1

        return randbaseval

    def _obfuscate(self, file_paths, mode=None):
        # ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        ip_pattern = r'\b[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b'

        if self._lang == 'py':
            obfuscator = obfuscate_string_py
            obf_modes = ['bytearray', 'base64', 'base32', 'base16', 'hex', 'split_inline', 'rand_base_pad']
            lexer = PythonLexer()
        else:
            obfuscator = obfuscate_string_js
            obf_modes = ['bytearray', 'base64', 'hex', 'split_inline', 'rand_base_pad']
            lexer = JavascriptLexer()

        assert mode is None or (isinstance(mode, str) and mode in obf_modes)
        if mode is None:
            mode = random.choice(obf_modes)

        for file_path in file_paths:
            try:
                with open(file_path, "r") as file:
                    file_content = file.read()
            except Exception:
                continue
            else:
                tokens = lexer.get_tokens(file_content)
                for token_type, token_value in tokens:
                    if "Token.Literal.String" in str(token_type):
                        for match_ip in re.finditer(ip_pattern, token_value):
                            if not self._check_ip(match_ip.group()):
                                continue

                            if mode == 'rand_base_pad':
                                obfuscated_ip = self._obf_rand_base_pad(match_ip.group())
                            else:
                                obfuscated_ip = obfuscator(match_ip.group(), mode)

                            file_content = re.sub(re.escape(match_ip.group()), obfuscated_ip, file_content)

            with open(file_path, "w") as file:
                file.write(file_content)


class ObfuscateIPsMetadata(ObfuscateIPs):
    def __call__(self, package):
        self._obfuscate(package.metadata_path)


class ObfuscateBase64Chunks(Transformation):
    def __call__(self, package):
        self._obfuscate(package.code_path)

    def _obfuscate(self, file_paths, mode=None):
        base64_patterns = [
            # r'(?:[A-Za-z0-9+\/]{4})*(?:[A-Za-z0-9+\/]{2}==|[A-Za-z0-9+\/]{3}=|[A-Za-z0-9+\/]{4})',
            (r"b64decode\s*\(\s*b?('|\")((?:[A-Za-z0-9\+\/]{4})*(?:[A-Za-z0-9\+\/]{2}==|[A-Za-z0-9\+\/]{3}=|[A-Za-z0-9\+\/]{4}))\1\s*\)", 2),
            (r'powershell.+?-EncodedCommand\s+((?:[A-Za-z0-9\+\/]{4})*(?:[A-Za-z0-9\+\/]{2}==|[A-Za-z0-9\+\/]{3}=|[A-Za-z0-9\+\/]{4}))', 1)
        ]
        if self._lang == 'py':
            obfuscator = obfuscate_string_py
            obf_modes = ['bytearray', 'base32', 'base16', 'hex', 'split_inline']
        else:
            obfuscator = obfuscate_string_js
            obf_modes = ['bytearray', 'hex', 'split_inline']

        assert mode is None or (isinstance(mode, str) and mode in obf_modes)
        if mode is None:
            mode = random.choice(obf_modes)

        for file_path in file_paths:
            try:
                with open(file_path, "r") as file:
                    file_content = file.read()
            except Exception:
                continue
            else:
                for base64_pattern, group_idx in base64_patterns:
                    for match in re.finditer(base64_pattern, file_content):
                        if not check_base64(match.group(group_idx)):
                            continue
                        obfuscated = obfuscator(match.group(group_idx), mode)

                        # start, end = re.search(match, file_content).span()[0]
                        # file_content = file_content[:start] + payload + file_content[:end]
                        try:
                            file_content = re.sub(match.group(group_idx), obfuscated, file_content)
                        except Exception:
                            # print(f"Failed to replace {match.group(group_idx)} in {file_path}")
                            continue

                with open(file_path, "w") as file:
                    file.write(file_content)


class ObfuscateBase64ChunksMetadata(ObfuscateBase64Chunks):
    def __call__(self, package):
        self._obfuscate(package.metadata_path)


class ObfuscateURLs(Transformation):
    def __call__(self, package):
        self._obfuscate(package.code_path)

    def _obfuscate(self, file_paths, mode=None):
        # Based on this blog post by Mandiant: https://www.mandiant.com/resources/blog/url-obfuscation-schema-abuse
        url_pattern = r'\b(?:https|http|hxxp):\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)'
        # url_pattern = r"""((?:(?:https|http|hxxp|ssh|ftp|sftp|ws|wss|dns|file|git|jni|imap|ldap|ldaps|nfs|smb|smbs|telnet|udp|vnc):(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|org|uk)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|uk|ac)\b/?(?!@)))"""
        if self._lang == 'py':
            obfuscator = obfuscate_string_py
            obf_modes = ['bytearray', 'base64', 'base32', 'base16', 'hex', 'split_inline']
            lexer = PythonLexer()
        else:
            obfuscator = obfuscate_string_js
            obf_modes = ['bytearray', 'base64', 'hex', 'split_inline']
            lexer = JavascriptLexer()
        
        assert mode is None or (isinstance(mode, str) and mode in obf_modes)
        if mode is None:
            mode = random.choice(obf_modes)

        for file_path in file_paths:
            try:
                with open(file_path, "r") as file:
                    file_content = file.read()
            except Exception:
                continue
            else:
                tokens = lexer.get_tokens(file_content)
                for token_type, token_value in tokens:
                    if "Token.Literal.String" in str(token_type):
                        for match in re.finditer(url_pattern, token_value):
                            obfuscated = obfuscator(match.group(), mode)

                            # start, end = re.search(match, file_content).span()[0]
                            # file_content = file_content[:start] + payload + file_content[:end]
                            file_content = re.sub(re.escape(match.group()), obfuscated, file_content)

                with open(file_path, "w") as file:
                    file.write(file_content)


class ObfuscateURLsMetadata(ObfuscateURLs):
    def __call__(self, package):
        self._obfuscate(package.metadata_path)


class ObfuscateCmds(Transformation):

    def __call__(self, package):
        self._obfuscate(package.code_path)

    def _obfuscate(self, file_paths, mode=None):
        lexer = PythonLexer() if self._lang == 'py' else JavascriptLexer()

        if self._lang == 'py':
            obfuscator = obfuscate_string_py
            obf_modes = ['bytearray', 'base64', 'base32', 'base16', 'hex', 'split_inline']
            lexer = PythonLexer()
        else:
            obfuscator = obfuscate_string_js
            obf_modes = ['bytearray', 'base64', 'hex', 'split_inline']
            lexer = JavascriptLexer()

        assert mode is None or (isinstance(mode, str) and mode in obf_modes)
        if mode is None:
            mode = random.choice(obf_modes)

        for file_path in file_paths:
            try:
                with open(file_path, "r") as file:
                    file_content = file.read()
            except Exception:
                continue

            tokens = lexer.get_tokens(file_content)
            for token_type, token_value in tokens:
                if "Token.Literal.String" in str(token_type):
                    num_strings += 1

                    # counter_matches = 0
                    # for cmd in ObfuscateCmds.SUSPCIOUS_TOKENS_CMDS:
                    #     if re.search(cmd, token_value):
                    #         counter_matches += 1
                    # if counter_matches > 0:
                    #     obfuscated = obfuscator(token_value, mode)
                    #     file_content = re.sub(re.escape(token_value), obfuscated, file_content)

                    for cmd in SUSPCIOUS_TOKENS_CMDS:
                        match = re.search(cmd, token_value)
                        if match:
                            # if there is a match, obfuscate the entire string
                            obfuscated = obfuscator(token_value, mode)
                            file_content = re.sub(re.escape(token_value), obfuscated, file_content)
                            break


class ObfuscateCmdsMetadata(ObfuscateCmds):
    def __call__(self, package):
        self._obfuscate(package.metadata_path)


class ObfuscateFSPaths(Transformation):
    def __call__(self, package):
        self._obfuscate(package.code_path)

    def _obfuscate(self, file_paths, mode=None):
        fs_paths_patterns = [
            r"([^\\/:*?\"'<>|\r\n][\\/]?)+([^\\/:*?\"'<>|\r\n])*\.?[a-z]*",   # Windows and Unix relative paths
            r"([a-zA-Z]:)?[\\/](?:[^\\/:*?\"'<>|\r\n\t]+[\\/])*[^\\/:*?\"'<>|\r\n\t]*" # Windows absolute paths
        ]
        if self._lang == 'py':
            obfuscator = obfuscate_string_py
            obf_modes = ['bytearray', 'base64', 'base32', 'base16', 'hex', 'split_inline']
            lexer = PythonLexer()
        else:
            obfuscator = obfuscate_string_js
            obf_modes = ['bytearray', 'base64', 'hex', 'split_inline']
            lexer = JavascriptLexer()

        assert mode is None or (isinstance(mode, str) and mode in obf_modes)
        if mode is None:
            mode = random.choice(obf_modes)

        for file_path in file_paths:
            try:
                with open(file_path, "r") as file:
                    file_content = file.read()
            except Exception:
                continue
            else:
                tokens = lexer.get_tokens(file_content)
                for token_type, token_value in tokens:
                    if "Token.Literal.String" in str(token_type):
                        for pattern in fs_paths_patterns:
                            match = re.match(pattern, token_value)
                            if match:
                                obfuscated = obfuscator(match.group(), mode)

                                file_content = re.sub(re.escape(match.group()), obfuscated, file_content)

                with open(file_path, "w") as file:
                    file.write(file_content)


class ObfuscateFSPathsMetadata(ObfuscateFSPaths):
    def __call__(self, package):
        self._obfuscate(package.metadata_path)


class ObfuscateAPIsPy(Transformation):
    """
    Example of patterns based on 'os' module and 'popen' function:

    os.popen(  -> DEFAULT
    os.popen.__call__(
    os.__dict__['popen'](
    os.__dict__['popen'].__call__(
    os.__getattribute__('popen')(
    os.__getattribute__('popen').__call__(
    __import__('os').popen(
    __import__('os').popen.__call__(
    __import__('os').__dict__['popen'](
    __import__('os').__dict__['popen'].__call__(
    __import__('os').__getattribute__('popen')(
    __import__('os').__getattribute__('popen').__call__(
    getattr(os, 'popen')(
    getattr(os, 'popen').__call__(
    getattr(__import__('os'), 'popen')(
    getattr(__import__('os'), 'popen').__call__(
    """
    def __call__(self, package):
        self._obfuscate(package.code_path)

    def _obfuscate(self, file_paths, mode='swap'):

        assert mode in ['swap', 'new_var', 'monkey_patch', 'monkey_patch_new_file']

        api_patterns = {
            "pyperclip": ["paste", "copy"],
            "pandas": ["read_clipboard"],
            "builtins": ["eval", 'exec'],
            "subprocess": ["getoutput", "call", "check_output", "run", "Popen"],
            "os": ["popen", "system", "posix_spawn", "posix_spawnp", "getenv", "chmod"] + [f"exec{mode}{submode}" for mode in ['l', 'v'] for submode in ['', 'e', 'p', 'pe']] + [f"spawn{mode}{submode}" for mode in ['l', 'v'] for submode in ['', 'e', 'p', 'pe']],
            "urllib": ["urlopen"],
            "request": ["urlopen", "Request"],  # urllib.request
            "requests": ["get", "post", "put", "patch", "delete", "request", "Session"],
            "socket": ["socket", "gethostname", "connect", "send", "sendfile", "sendto", "sendmsg", "sendall", "recv", "recvfrom", "recvmsg"]
        }

        obfuscations_patterns = []
        for select_module in ["__import__('{module}')."]:  # ["{module}.", "__import__('{module}')."]:
            for select_func in ["{func}", "__dict__['{func}']", "__getattribute__('{func}')"]:
                for call_func in ["(", ".__call__("]:
                    obfuscations_patterns.append(f"{select_module}{select_func}{call_func}")
        # obfuscations_patterns = [f"{select_module}{select_func}{call_func}" for select_module in ["{module}.", "__import__('{module}')."] for select_func in ["{func}", "__dict__['{func}']", "__getattribute__('{func}')"] for call_func in ["(", ".__call__("]]

        for select_module in ["{module}", "__import__('{module}')"]:
            for call_func in ["(", ".__call__("]:
                obfuscations_patterns.append("getattr({module}, '{func_fmt_name}'){call}".format(module=select_module, func_fmt_name="{func}", call=call_func))
        # obfuscations_patterns += ["getattr({module}, '{func_fmt_name}'){call}".format(module=select_module, func_fmt_name="{func}", call=call_func) for select_module in ["{module}", "__import__('{module}')"] for call_func in ["(", ".__call__("]]

        # handle here other obfuscation patterns related to eval() and exec()
        for select_builtins in ["['__builtins__']", ".get('builtins')"]:
            for select_func in ["{func}", "__dict__['{func}']", "__getattribute__('{func}')"]:
                for call_func in ["(", ".__call__("]:
                    obfuscations_patterns.append(f"globals(){select_builtins}.{select_func}{call_func}")

        for select_builtins in ["['__builtins__']", ".get('builtins')"]:
            for call_func in ["(", ".__call__("]:
                obfuscations_patterns.append("getattr(globals(){builtins}, '{func_fmt_name}'){call}".format(builtins=select_builtins, func_fmt_name="{func}", call=call_func))

        # repl_pattern = random.choice(obfuscations_patterns)
        repl_pattern = "getattr({module}, '{func}').__call__("

        for file_path in file_paths:
            try:
                with open(file_path, "r") as file:
                    file_content = file.read()
            except Exception:
                continue
            else:
                for module, functions in api_patterns.items():
                    for func in functions:
                        if mode == 'new_var':
                            repl_var = "_".join(c for c in module + func)
                            repl = f"{module}.{func}={repl_var};{repl_var}("
                            # replace all the occurrencies of <module>.<function>(
                            file_content = re.sub(r"\b{module}\s*\.\s*{func}\s*\(".format(module=module, func=func), repl, file_content)
                        elif mode == 'swap':
                            # replace all the occurrencies of <module>.<function>(
                            file_content = re.sub(r"\b{module}\s*\.\s*{func}\s*\(".format(module=module, func=func), repl_pattern.format(module=module, func=func), file_content)
                        
                        # otherwise, first check if the module has been imported and then replace all the occurrencies of <function>(
                        # NOTE: experiemntal: may lead to false positives if there is a function with the same name of the imported module
                        is_module_imported = re.search(r"\bimport\s+(\w+,\s*)*({module})".format(module=module), file_content)
                        if is_module_imported:
                            if mode == 'new_var':
                                repl_var = "_".join(c for c in func)
                                repl = f"{func}={repl_var};{repl_var}("
                                re.sub(r"(?<!def\s)\b{func}\s*\(".format(func=func), repl, file_content)
                            elif mode == 'swap':
                                re.sub(r"(?<!def\s)\b{func}\s*\(".format(func=func), repl_pattern.format(module=module, func=func), file_content)
            
            with open(file_path, 'w') as file:
                file.write(file_content)


class ObfuscateAPIsPyMetadata(ObfuscateAPIsPy):
    def __call__(self, package):
        self._obfuscate(package.metadata_path)


class ObfuscateInstallMetadata(Transformation):
    def __call__(self, package):
        self._obfuscate(package.metadata_path)

    def _obfuscate(self, file_paths, mode='hex'):
        if self._lang == 'py':
            obfuscator = obfuscate_string_py
            obf_modes = ['bytearray', 'base64', 'base32', 'base16', 'hex', 'split_inline']
            lexer = PythonLexer()
            filename = 'setup.py'
            install_patterns = ['install']
        else:
            obfuscator = obfuscate_string_js
            obf_modes = ['bytearray', 'base64', 'hex', 'split_inline']
            lexer = JavascriptLexer()
            filename = 'package.json'
            install_patterns = ['preinstall', 'install',
                                      # 'preprepare', 'prepare', 'postprepare', 'prepublish'
                                      'postinstall']

        assert mode in obf_modes

        if isinstance(file_paths, list) and len(file_paths) == 1 and file_paths[0].endswith(filename):
            try:
                with open(file_paths[0], 'r', encoding='utf-8') as file:
                    file_content = file.read()
            except Exception:
                return

            tokens = lexer.get_tokens(file_content)
            new_content = ""
            for token_type, token_value in tokens:
                if "Token.Literal.String" in str(token_type):
                    for pattern in install_patterns:
                        if re.match(pattern, token_value):
                            token_value = obfuscator(token_value, mode)
                new_content += token_value

            with open(file_paths[0], 'w') as file:
                file.write(new_content)


class InjectNewFile(Transformation):
    def __call__(self, package, file_ext='h'):

       # TODO: Improve this transformation by selecting the most common file extensions based in benign packages
        text_extensions = {'bat', 'c', 'conf', 'cpp', 'css', 'html', 'java', 'js', 'json', 'm4v', 'markdown', 'md',
                           'py', 'rb', 'rst', 'sh', 'sql', 'toml', 'txt', 'xml', 'yaml', 'yml', 'crt', 'key', 'pem',
                           'h', 'c', 'cpp', 'cc', 'cxx', 'hxx', 'hpp', 'h++', 'cxx', 'c++'}

        # filename_candidates = ['_tmp', '_test', '_foo', '_tmp_', '_test_', '_foo_']
        # filename = random.choice(filename_candidates)
        assert file_ext in text_extensions or file_ext is None
        if file_ext is None:
            file_ext = random.choice(list(text_extensions))

        find_new_string = True
        while find_new_string:
            filename = random_string()

            # Combine with the output directory
            pkg_folders = glob.glob(os.path.join(package.path, "**/"), recursive=True)
            if not pkg_folders:
                return
            out_dir = random.choice(pkg_folders)
            file_path = os.path.join(out_dir, f"{filename}.{file_ext}")
            if not os.path.isfile(file_path):
                find_new_string = False

        # Determine if the file is text or binary based on the extension
        is_text_file = file_ext in text_extensions

        # Create an empty file
        mode = 'w' if is_text_file else 'wb'
        with open(file_path, mode):
            pass


class InjectPlus(Transformation):
    def __call__(self, package):
        self._base_path = package.path
        self._inject_plus(package.code_path)

    def _inject_plus(self, file_paths, mode='use_new_func', create_new_file=True, counts=5):
        assert mode in ['use_new_func', 'split_number']
        if not file_paths:
            if create_new_file:
                target_file = create_new_file_py(self._base_path)
            else:
                return
        else:
            target_file = random.choice(file_paths)

        try:
            with open(target_file, 'r') as file:
                file_content = file.read()
        except Exception:
            return

        lexer = PythonLexer() if self._lang == 'py' else JavascriptLexer()

        if mode == "split_number":
            new_content = ""
            tokens = list(lex(file_content, lexer))
            for token_type, token_value in tokens:
                if str(token_type) == Token.Literal.Number:
                    offset = random.randint(1, 10)
                    token_value = "{} + {}".format(int(token_value) - offset, offset)
                new_content += token_value

                with open(target_file, 'w') as file:
                    file.write(new_content)

        elif mode == 'use_new_func':
            # check if the name for the new function to be created is already taken
            tokens = list(lex(file_content, lexer))

            func_names_file = set()
            for token_type, token_value in tokens:
                if str(token_type) == Token.Name.Function:
                    func_names_file.add(token_value)

            # func_name = random_string()
            # while func_name in func_names_file:
            #     func_name = random_string()
            # values = [random.randint(1, 100) for _ in range(counts)]
            # with open(target_file, 'a') as file:
            #     file.write(f"\ndef {func_name}():\n\treturn {' + '.join(map(str, values))}")
            with open(target_file, 'a') as file:
                for _ in range(counts):
                    func_name = random_string()
                    while func_name in func_names_file:
                        func_name = random_string()
                    values = [random.randint(1, 100) for _ in range(2)]
                    if self._lang == 'py':
                        file.write(f"\ndef {func_name}():\n\treturn {' + '.join(map(str, values))}")
                    else:
                        file.write(f"\nfunction {func_name}() {{\n\treturn {' + '.join(map(str, values))};\n}}")


class InjectPlusMetadata(InjectPlus):
    def __call__(self, package):
        super()._inject_plus(package.metadata_path, create_new_file=False)


class InjectEq(Transformation):
    def __call__(self, package):
        self._base_path = package.path
        self._inject_eq(package.code_path)

    def _inject_eq(self, file_paths, create_new_file=True, counts=5):
        if not file_paths:
            if create_new_file:
                target_file = create_new_file_py(self._base_path)
            else:
                return
        else:
            target_file = random.choice(file_paths)

        try:
            with open(target_file, 'r') as file:
                file_content = file.read()
        except Exception:
            return

        lexer = PythonLexer() if self._lang == 'py' else JavascriptLexer()

        # check if the name for the new function to be created is already taken
        tokens = list(lex(file_content, lexer))

        func_names_file = set()
        for token_type, token_value in tokens:
            if str(token_type) == Token.Name.Function:
                func_names_file.add(token_value)

        func_name = random_string()
        while func_name in func_names_file:
            func_name = random_string()
        
        # code = ""
        # for _ in range(counts):
        #     var_name = random_string()
        #     rand_int = random.randint(1, 100)
        #     code += f"\n\t{var_name} = {rand_int}"
        # with open(target_file, 'a') as file:
        #     file.write(f"\ndef {func_name}():{code}\n")
        with open(target_file, 'a') as file:
            for _ in range(counts):
                var_name = random_string()
                rand_int = random.randint(1, 100)
                if self._lang == 'py':
                    file.write(f"\ndef {func_name}():\n\t{var_name} = {rand_int}")
                else:
                    file.write(f"\nfunction {func_name}() {{\n\tconst {var_name} = {rand_int};\n}}")


class InjectEqMetadata(InjectEq):
    def __call__(self, package):
        super()._inject_eq(package.metadata_path, create_new_file=False)


class InjectBrackets(Transformation):
    def __call__(self, package):
        self._base_path = package.path
        self._inject_brackets(package.code_path)

    def _inject_brackets(self, file_paths, create_new_file=True, counts=5):
        if not file_paths:
            if create_new_file:
                target_file = create_new_file_py(self._base_path)
            else:
                return
        else:
            target_file = random.choice(file_paths)

        try:
            with open(target_file, 'r') as file:
                file_content = file.read()
        except Exception:
            return

        lexer = PythonLexer() if self._lang == 'py' else JavascriptLexer()
        tokens = list(lex(file_content, lexer))
        # check if the name for the new function to be created is already taken
        func_names_file = set()
        for token_type, token_value in tokens:
            if str(token_type) == Token.Name.Function:
                func_names_file.add(token_value)

        # func_name = random_string()
        # while func_name in func_names_file:
        #     func_name = random_string()
        # values = [[random.randint(1, 100) for _ in range(random.randint(1, 8))] for _ in range(counts)]
        # code = " + ".join(map(str, values))
        # with open(target_file, 'a') as file:
        #     file.write(f"\ndef {func_name}():\n\treturn {code}")
        with open(target_file, 'a') as file:
            for _ in range(counts):
                func_name = random_string()
                while func_name in func_names_file:
                    func_name = random_string()
                values = [random.randint(1, 100) for _ in range(random.randint(1, 8))]
                if self._lang == 'py':
                    file.write(f"\ndef {func_name}():\n\treturn {values}")
                else:
                    file.write(f"\nfunction {func_name}() {{\n\treturn {values};\n}}")


class InjectBracketsMetadata(InjectBrackets):
    def __call__(self, package):
        super()._inject_brackets(package.metadata_path, create_new_file=False)


class InjectRandCode(Transformation):
    def __call__(self, package):
        self._base_path = package.path
        self._inject_code(package.code_path)
    
    def _generate_code_snippets(self):
        rand_func_name1 = random_string()
        rand_func_name2 = random_string()
        rand_var_name1 = random_string()
        rand_var_name2 = random_string()
        rand_var_name3 = random_string()
        rand_int = random.randint(1, 10)
        rand_string_val = random_string()
        rand_int_list = [random.randint(1, 100) for _ in range(random.randint(1, 8))]

        if self._lang == 'py':
            code_snippets = [
                f"\ndef {rand_func_name1}():\n\tdef {rand_func_name2}():\n\t\t{rand_var_name1} = '{rand_string_val}'\n\t{rand_var_name2} = {rand_int}\n\t{rand_var_name3} = {rand_int_list}; {rand_var_name2} += 1"
            ]
        else:
            code_snippets = [
                f"\nfunction {rand_func_name1}() {{\n\tfunction {rand_func_name2}() {{\n\t\tconst {rand_var_name1} = '{rand_string_val}';\n\t}}\n\tlet {rand_var_name2} = {rand_int};\n\tconst {rand_var_name3} = {rand_int_list};\n\t{rand_var_name2} += 1;\n}}"
            ]
        return code_snippets

    def _inject_code(self, file_paths, create_new_file=True, counts=5):
        if not file_paths:
            if create_new_file:
                target_file = create_new_file_py(self._base_path)
            else:
                return
        else:
            target_file = random.choice(file_paths)

        with open(target_file, 'a') as file:
            for _ in range(counts):
                code_snippets = self._generate_code_snippets()
                selected_snippet = random.choice(code_snippets)

                file.write(selected_snippet)


class InjectRandCodeMetadata(InjectRandCode):
    def __call__(self, package):
        super()._inject_code(package.metadata_path, create_new_file=False)


class InjectWhitespace(Transformation):
    def __call__(self, package, ws_char=' '):
        self._base_path = package.path
        self._inject_ws(package.code_path, ws_char)

    def _inject_ws(self, file_paths, ws_char, create_new_file=True, counts=10):
        if not file_paths:
            if create_new_file:
                target_file = create_new_file_py(self._base_path)
            else:
                return
        else:
            target_file = random.choice(file_paths)

        # basic approach: add ws chars at the end of the file
        with open(target_file, 'a') as file:
            file.write(ws_char * counts)

        ## NOTE: Experimental code to inject ws chars within the code
        # with open(target_file, 'r+') as file:
        #     file_content = file.read()
        #
        # lexer = PythonLexer() if self._lang == 'py' else JavascriptLexer()
        # tokens = list(lex(file_content, lexer))
        # # check if the name for the new function to be created is already taken
        # num_ws_added = 0
        # new_code = ""
        # for token_type, token_value in tokens:
        #     if ws_char == " ":
        #         if "Token.Literal" in str(token_type) or "Token.Keyword" in str(token_type) or \
        #                 str(token_type) == "Token.Operator" or \
        #                 ("Token.Punctuation" in str(token_type) and token_value in ['(', ')', '[', ']', '{', '}', ':', ',']) or \
        #                 ("Token.Whitespace" in str(token_type) and token_value in ['\n', '']):
        #             token_value += ws_char
        #             num_ws_added += 1
        #     elif ws_char == "\n":
        #         if "Token.Punctuation" in str(token_type) and token_value in ['(', ')', '[', ']', '{', '}', ':', ','] or \
        #                 ("Token.Whitespace" in str(token_type) and token_value in ['\n']):
        #             token_value += ws_char
        #             num_ws_added += 1
        #     elif ws_char == "\t":
        #         for token_type, token_value in tokens:
        #             if "Token.Operator" in str(token_type) and token_value in ['=', '+', '-', '*', '/'] or \
        #                     "Token.Punctuation" in str(token_type) and token_value in ['(', ')', '[', ']', '{', '}', ':', ','] or \
        #                     ("Token.Whitespace" in str(token_type) and token_value in [' ']):
        #                 token_value += ws_char
        #                 num_ws_added += 1
        #     new_code += token_value
        #
        # with open(target_file, 'w') as file:
        #     file.write(new_code)
        #     file.write(ws_char * (counts - num_ws_added))


class InjectSpace(InjectWhitespace):
    def __call__(self, package):
        self._base_path = package.path
        self._inject_ws(package.code_path, ws_char=' ')


class InjectSpaceMetadata(InjectWhitespace):
    def __call__(self, package):
        self._inject_ws(package.metadata_path, ws_char=' ', create_new_file=False)


class InjectNewline(InjectWhitespace):
    def __call__(self, package):
        self._base_path = package.path
        self._inject_ws(package.code_path, ws_char='\n')


class InjectNewlineMetadata(InjectWhitespace):
    def __call__(self, package):
        self._inject_ws(package.metadata_path, ws_char='\n', create_new_file=False)


class InjectTab(InjectWhitespace):
    def __call__(self, package):
        self._base_path = package.path
        self._inject_ws(package.code_path, ws_char='\t')


class InjectTabMetadata(InjectWhitespace):
    def __call__(self, package):
        self._inject_ws(package.metadata_path, ws_char='\t', create_new_file=False)


class InjectIP(Transformation):
    def __call__(self, package):
        self._base_path = package.path
        self._inject_ip(package.code_path)

    def _inject_ip(self, file_paths, create_new_file=True, counts=5):
        if not file_paths:
            if create_new_file:
                target_file = create_new_file_py(self._base_path)
            else:
                return
        else:
            target_file = random.choice(file_paths)

        try:
            with open(target_file, 'r') as file:
                file_content = file.read()
        except Exception:
            return

        lexer = PythonLexer() if self._lang == 'py' else JavascriptLexer()
        tokens = list(lex(file_content, lexer))
        # check if the function is already in the file
        func_names_file = set()
        for token_type, token_value in tokens:
            if str(token_type) == Token.Name.Function:
                func_names_file.add(token_value)

        with open(target_file, 'a') as file:
            for _ in range(counts):
                func_name = random_string()
                while func_name in func_names_file:
                    func_name = random_string()
                
                var_name = random_string()
                # ip = random.choice(['127.0.0.1', '192.168.1.255', '0.0.0.0'])
                ip = '{}.{}.{}.{}'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                if self._lang == 'py':
                    file.write(f"\ndef {func_name}():\n\t{var_name} = '{ip}'")
                else:
                    file.write(f"\nfunction {func_name}() {{\n\tconst {var_name} = '{ip}';\n}}")


class InjectIPMetadata(InjectIP):
    def __call__(self, package):
        super()._inject_ip(package.metadata_path, create_new_file=False)


class InjectURL(Transformation):
    def __call__(self, package):
        self._base_path = package.path
        self._inject_url(package.code_path)

    def _inject_url(self, file_paths, create_new_file=True, counts=5):
        if not file_paths:
            if create_new_file:
                target_file = create_new_file_py(self._base_path)
            else:
                return
        else:
            target_file = random.choice(file_paths)

        try:
            with open(target_file, 'r') as file:
                file_content = file.read()
        except Exception:
            return

        lexer = PythonLexer() if self._lang == 'py' else JavascriptLexer()
        tokens = list(lex(file_content, lexer))

        # check if the function is already in the file
        func_names_file = set()
        for token_type, token_value in tokens:
            if str(token_type) == Token.Name.Function:
                func_names_file.add(token_value)

        with open(target_file, 'a') as file:
            for _ in range(counts):
                func_name = random_string()
                while func_name in func_names_file:
                    func_name = random_string()
                
                var_name = random_string()
                urls_list = [
                    'https://www.google.com',
                    'https://www.youtube.com',
                    'https://www.facebook.com',
                    'https://www.wikipedia.org',
                    'https://www.amazon.com',
                    'https://www.baidu.com',
                    'https://www.openai.com',
                    'https://www.microsoft.com',
                    'https://www.ebay.com',
                    'https://www.quora.com',
                    'https://www.zoom.us',
                    'https://www.discord.com',
                    'https://www.msn.com',
                    'https://www.outlook.com',
                    'https://www.duckduckgo.com',
                    'https://www.linkedin.com',
                    'https://www.netflix.com'
                ]
                url = random.choice(urls_list)

                if self._lang == 'py':
                    file.write(f"\ndef {func_name}():\n\t{var_name} = '{url}'")
                else:
                    file.write(f"\nfunction {func_name}() {{\n\tconst {var_name} = '{url}';\n}}")


class InjectURLMetadata(InjectURL):
    def __call__(self, package):
        super()._inject_url(package.metadata_path, create_new_file=False)