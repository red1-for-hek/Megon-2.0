import re, unicodedata
def run(text):
    t = unicodedata.normalize('NFKC', str(text))
    t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', t)
    return re.sub(r'\s+', ' ', t).strip()
