import sys, json, glob
sys.path.insert(0, '.')
from config.settings import RAW_DIR

files = sorted(glob.glob(str(RAW_DIR / '**' / '*_run.json'), recursive=True))
print('Latest archive:', files[-1])
data = json.load(open(files[-1]))
md   = data.get('market_data', [])
live = [d for d in md if d.get('type') == 'market_data_live']
print('Live records in archive:', len(live))
for d in live:
    print(d['label'], d['value'], d['period'])