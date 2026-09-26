import re
def run(pred, gold):
    p = set(re.findall(r"[a-z0-9']+", str(pred).lower()))
    g = set(re.findall(r"[a-z0-9']+", str(gold).lower()))
    if not p or not g:
        return 0.0
    i = len(p & g)
    if not i:
        return 0.0
    pr, rc = i / len(p), i / len(g)
    return round(2 * pr * rc / (pr + rc), 4)
