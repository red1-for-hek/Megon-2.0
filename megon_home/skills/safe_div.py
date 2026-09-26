def run(a, b):
    try:
        b = float(b)
        return None if b == 0 else float(a) / b
    except Exception:
        return None
