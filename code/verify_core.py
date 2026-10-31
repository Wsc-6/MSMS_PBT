"""Check the core-source snapshot using only the Python standard library."""
from pathlib import Path
import ast
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parent

def main():
    with (ROOT/'SHA256SUMS.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        path=(ROOT/row['path']).resolve()
        if ROOT not in path.parents or not path.is_file():
            raise ValueError(f"Missing or invalid path: {row['path']}")
        if path.stat().st_size!=int(row['bytes']) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError(f"File differs from snapshot: {row['path']}")
    scripts=list(ROOT.rglob('*.py'))
    for path in scripts:ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    notebooks=list(ROOT.rglob('*.ipynb'))
    for path in notebooks:
        nb=json.loads(path.read_text(encoding='utf-8'))
        if nb.get('nbformat')!=4 or not isinstance(nb.get('cells'),list):
            raise ValueError(f"Invalid notebook: {path.name}")
        for cell in nb['cells']:
            if cell['cell_type']=='code' and (cell.get('outputs') or cell.get('execution_count') is not None):
                raise ValueError(f"Unexpected execution output: {path.name}")
    print(f"PASS: {len(rows)} hashes; {len(scripts)} Python scripts; {len(notebooks)} source notebooks.")

if __name__=='__main__':main()
