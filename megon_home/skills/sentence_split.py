import re
def run(text):
    t = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|St|No|vs|etc|Inc|Ltd)\.', r'\1<DOT>', str(text))
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', re.sub(r'\s+', ' ', t).strip())
    return [p.replace('<DOT>', '.').strip() for p in parts if len(p.strip()) > 2]
