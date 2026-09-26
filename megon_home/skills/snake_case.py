import re


def run(x):
    return str('_'.join(str(i) for i in str(x).split())).lower()
