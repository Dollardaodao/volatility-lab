"""Hash the explicitly allowed public release files, excluding local data."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    files=[ROOT/n for n in ['README.md','.gitignore','.env.example','pyproject.toml','requirements.lock.txt']]
    files+=list(ROOT.glob('*_plan*.json'))
    for folder in ['volatility_lab','tests','scripts']:files+=list((ROOT/folder).glob('*.py'))
    files+=list((ROOT/'docs').rglob('*.md'))+list((ROOT/'docs').rglob('*.png'))
    files+=list((ROOT/'reports').rglob('*.csv'))+list((ROOT/'reports').rglob('*.json'))
    files+=list((ROOT/'.github/workflows').glob('*.yml'))
    target=ROOT/'reports/publication_manifest.json'
    files=sorted(set(p for p in files if p!=target))
    result={'scope':'Public code, aggregate research outputs, reports and provenance only; no raw market data.',
        'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Wrote release hashes for {len(files)} explicitly allowed public files.')


if __name__=='__main__':main()
