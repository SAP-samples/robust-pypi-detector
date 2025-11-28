import sys
sys.path.insert(0, "../")
from features_obfuscation_soa_detector import detect_api_obfuscation
import os


def test_detect_api_obfuscation():
    res = detect_api_obfuscation([os.path.join(os.path.dirname(__file__), "api_obfuscation_example.py")])
    assert res == 14, f"Expected 14 obfuscated API calls, but got {res}"


if __name__ == "__main__":
    test_detect_api_obfuscation()