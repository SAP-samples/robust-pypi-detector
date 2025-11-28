import re


def analyze_package_obfuscation(package_files, patterns):
    counter = 0
    for file in package_files:
        try:
            with open(file, 'r') as f:
                code = f.read()
        except Exception as e:
            continue
        for pattern in patterns:
            counter += len(list(re.finditer(pattern, code)))

    return counter


def detect_string_obfuscation_base64(package_files):
    patterns = [
        # r"base64\.b64decode\s*\(.*\)",
        # r"__import__\(('|\")base64\1\)\.b64decode\s*\(.*\)",
        r"b64decode\s*\(.*\)",
        r"b64encode\s*\(.*\)"
    ]

    return analyze_package_obfuscation(package_files, patterns)


def detect_string_obfuscation_base32(package_files):
    patterns = [
        # r"base64\.b32decode\s*\(.*\)",
        # r"__import__\(('|\")base32\1\)\.b32decode\s*\(.*\)",
        r"b32decode\s*\(.*\)",
        r"b32encode\s*\(.*\)"
    ]

    return analyze_package_obfuscation(package_files, patterns)


def detect_string_obfuscation_base16(package_files):
    patterns = [
        # r"base64\.b16decode\s*\(.*\)",
        # r"__import__\(('|\")base16\1\)\.b16decode\s*\(.*\)",
        r"b16decode\s*\(.*\)",
        r"b16encode\s*\(.*\)"
    ]

    return analyze_package_obfuscation(package_files, patterns)


def detect_string_obfuscation_baseXX(package_files):
    patterns_b64 = [
        # r"base64\.b64decode\s*\(.*\)",
        # r"__import__\(('|\")base64\1\)\.b64decode\s*\(.*\)",
        r"b64decode\s*\(.*\)",
        r"b64encode\s*\(.*\)"
    ]

    patterns_b32 = [
        # r"base64\.b32decode\s*\(.*\)",
        # r"__import__\(('|\")base32\1\)\.b32decode\s*\(.*\)",
        r"b32decode\s*\(.*\)",
        r"b32encode\s*\(.*\)"
    ]

    patterns_b16 = [
        # r"base64\.b16decode\s*\(.*\)",
        # r"__import__\(('|\")base16\1\)\.b16decode\s*\(.*\)",
        r"b16decode\s*\(.*\)",
        r"b16encode\s*\(.*\)"
    ]

    patterns_b85 = [
        # r"base64\.b85decode\s*\(.*\)",
        # r"__import__\(('|\")base85\1\)\.b85decode\s*\(.*\)",
        r"b85decode\s*\(.*\)",
        r"b85encode\s*\(.*\)"
    ]

    patterns = patterns_b64 + patterns_b32 + patterns_b16  + patterns_b85

    return analyze_package_obfuscation(package_files, patterns)


def detect_string_obfuscation_hex(package_files):
    patterns = [
        r"\bbytes\.fromhex\s*\(.*\)",
        r"\bhex\s*\(.*\)",
        r"(\\x[0-9a-fA-F]{2})+",
    ]

    return analyze_package_obfuscation(package_files, patterns)


def detect_string_obfuscation_mixed_encoding(package_files):
    patterns = [
        r"(\\x[0-9a-fA-F]{2}|\\[0-7]{3})+",
    ]

    return analyze_package_obfuscation(package_files, patterns)


def detect_string_obfuscation_bytearray(package_files):
    patterns = [
        r"\bbytes\s*\(.*\)"
    ]

    return analyze_package_obfuscation(package_files, patterns)


def detect_string_obfuscation_rot13(package_files):
    patterns = [
        r"codecs\.encode\s*\(.*\, ('|\")rot_13\1\)",
    ]

    return analyze_package_obfuscation(package_files, patterns)


def detect_string_obfuscation_splitting(package_files):
    patterns = [
        # r"('|\")[\w\s]+\1(\s*\+\s*('|\")[\w\s]+\3)+",
        r"('|\")[\w\s]*\1(\s*\+\s*('|\")[\w\s]*\3)+",
        # r"[a-zA-Z_]\w*(\s*\+\s*[a-zA-Z_]\w*)+",   # regex to be improved because we get stucked in some cases
        # r"join\s*\(\s*\[\s*('|\")[\w\s]+\1(\s*,\s*('|\")[\w\s]+\3)+\s*\]\s*\)",
        r"\.join\((\[|map)"
        # r"\.join\(.*\)"
    ]

    return analyze_package_obfuscation(package_files, patterns)


def detect_string_obfuscation_xor(package):
    patterns = [
        r"\bchr\s*\(.*\s*\^\s*.*\)",
        r"\bord\s*\(.*\s*\^\s*.*\)"
    ]

    return analyze_package_obfuscation(package, patterns)


def detect_api_obfuscation(package_files):
    obfuscations_patterns = []
    for select_module in [r"\w+\.", r"__import__\('\w+'\)\."]:
        for select_func in [r"(?!__)\w+", r"__dict__\['\w+'\]", r"__getattribute__\('\w+'\)"]:
            for call_func in [r"\(", r"\.__call__\("]:
                if select_module == r"\w+\." and select_func == r"\w+" and call_func == r"\(":
                    continue  # skip the non-obfuscated pattern
                obfuscations_patterns.append(f"{select_module}{select_func}{call_func}")
    # obfuscations_patterns = [f"{select_module}{select_func}{call_func}" for select_module in ["\w+.", "__import__('\w+')."] for select_func in ["\w+", "__dict__['\w+']", "__getattribute__('\w+')"] for call_func in ["(", ".__call__("]]

    for select_module in [r"\w+", r"__import__\('\w+'\)"]:
        for call_func in [r"\(", r"\.__call__\("]:
            obfuscations_patterns.append(r"getattr\({module}, '{func_fmt_name}'\){call}".format(module=select_module, func_fmt_name=r"\w+", call=call_func))
    # obfuscations_patterns += ["getattr(\w+, '{func_fmt_name}'){call}".format(module=select_module, func_fmt_name="\w+", call=call_func) for select_module in ["\w+", "__import__('\w+')"] for call_func in ["(", ".__call__("]]

    # handle here other obfuscation patterns related to eval() and exec()
    for select_builtins in [r"\['__builtins__'\]", r".get\('builtins'\)"]:
        for select_func in [r"\w+", r"__dict__\['\w+'\]", r"__getattribute__\('\w+'\)"]:
            for call_func in [r"\(", r"\.__call__\("]:
                obfuscations_patterns.append(f"globals(){select_builtins}.{select_func}{call_func}")

    for select_builtins in [r"\['__builtins__'\]", r".get\('builtins'\)"]:
        for call_func in [r"\(", r"\.__call__\("]:
            obfuscations_patterns.append(r"getattr\(globals\(\){builtins}, '{func_fmt_name}'\){call}".format(builtins=select_builtins, func_fmt_name=r"\w+", call=call_func))

    # detection_patterns = []
    # api_list = [SENSITIVE_API_NETWORK, SENSITIVE_API_FILESYSTEM, SENSITIVE_API_HOSTINFO, SENSITIVE_API_CMD_EXEC, SENSITIVE_API_ENCODING, SENSITIVE_API_CODE_EXEC]
    # for api_patterns in api_list:
    #     for base, funcs in api_patterns.items():
    #         if not funcs:
    #             continue
    #         for func in funcs:
    #             for pattern in obfuscations_patterns:
    #                 detection_patterns.append(pattern.format(module=base, func=func))

    obfuscations_patterns = [r"{}".format(pattern.replace("'", "('|\")")) for pattern in obfuscations_patterns]

    return analyze_package_obfuscation(package_files, obfuscations_patterns)