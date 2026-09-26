import re
def run(a, b):
    A = set(re.findall(r'[a-z0-9]+', str(a).lower()))
    B = set(re.findall(r'[a-z0-9]+', str(b).lower()))
    if not A and not B:
        return 1.0
    u = A | B
    return round(len(A & B) / len(u), 4) if u else 0.0
