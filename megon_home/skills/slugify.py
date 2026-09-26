import re


def run(x):
    return str('-'.join(str(i) for i in str(x).split())).lower()
