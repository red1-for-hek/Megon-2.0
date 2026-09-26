import re
def run(text):
    return [float(m.replace(',', '')) for m in
            re.findall(r'\d[\d,]*(?:\.\d+)?', str(text))]
