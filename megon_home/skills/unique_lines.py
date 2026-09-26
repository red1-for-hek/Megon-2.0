import re


def run(x):
    return chr(10).join(str(i) for i in list(dict.fromkeys(str(x).split())))
