import numpy as np

# Minimal element radii (van der Waals) and colors RGB
EL_VDW = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.80, 'S': 1.80,
}
EL_COLOR = {
    'H': (0.9,0.9,0.9), 'C': (0.2,0.2,0.2), 'N': (0.1,0.2,0.8), 'O': (0.8,0.1,0.1),
    'F': (0.4,0.9,0.4), 'P': (1.0,0.6,0.2), 'S': (1.0,0.9,0.1),
}

def load_xyz(path:str):
    with open(path,'r',encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return []
    try:
        n = int(lines[0].split()[0])
        start = 2
    except Exception:
        n = len(lines)
        start = 0
    atoms = []
    for ln in lines[start:start+n]:
        parts = ln.split()
        if len(parts) < 4: continue
        el = parts[0].capitalize()
        x,y,z = map(float, parts[1:4])
        r = EL_VDW.get(el, 1.5)
        col = EL_COLOR.get(el, (0.7,0.7,0.7))
        atoms.append({'el':el, 'pos': np.array([x,y,z], np.float32), 'r': float(r), 'color': col})
    return atoms
