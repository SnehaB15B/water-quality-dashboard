# utils/is_water_safe.py

def simple_is_water_safe(ph, turbidity, temp):
    """
    Simple rule-based check using (ph, turbidity, temp).
    Returns: safety (1 safe, 0 unsafe)
    """
    if 6.5 <= ph <= 8.5 and turbidity <= 5 and 10 <= temp <= 35:
        return 1
    return 0


def advanced_is_water_safe(row):
    """
    Advanced rule-based check using keys:
    'pH', 'turbidity', 'chlorine', 'ecoli', 'nitrates'
    row can be dict-like or pandas Series.
    Returns: (safe (bool), reasons (list of strings))
    """
    reasons = []
    safe = True

    # Accept either lowercase/uppercase keys: try common variants
    def get(k):
        # try a few common column names
        for key in (k, k.lower(), k.upper(), k.capitalize()):
            if key in row and row[key] is not None:
                return row[key]
        return None

    ph = get('pH')
    turbidity = get('turbidity')
    chlorine = get('chlorine')
    ecoli = get('ecoli') or get('E.coli') or get('Total Coliform (MPN/100ml)')
    nitrates = get('nitrates')

    if ph is not None:
        try:
            ph = float(ph)
            if not (6.5 <= ph <= 8.5):
                safe = False
                reasons.append("pH out of range")
        except:
            reasons.append("pH invalid")

    if turbidity is not None:
        try:
            turbidity = float(turbidity)
            if turbidity >= 5:
                safe = False
                reasons.append("high turbidity")
        except:
            reasons.append("turbidity invalid")

    if chlorine is not None:
        try:
            chlorine = float(chlorine)
            if chlorine < 0.2:
                safe = False
                reasons.append("low chlorine residual")
        except:
            reasons.append("chlorine invalid")

    if ecoli is not None:
        try:
            ecoli = float(ecoli)
            if ecoli > 0:
                safe = False
                reasons.append("microbial contamination (E. coli / Coliforms)")
        except:
            reasons.append("ecoli invalid")

    if nitrates is not None:
        try:
            nitrates = float(nitrates)
            if nitrates >= 50:
                safe = False
                reasons.append("high nitrates")
        except:
            reasons.append("nitrates invalid")

    return int(safe), reasons
