def run(text):
    seen, out = set(), []
    for ln in str(text).splitlines():
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return '\n'.join(out)
