import re


def run(x):
    return len(re.findall(r'[A-Za-z0-9]+', str(x)))
