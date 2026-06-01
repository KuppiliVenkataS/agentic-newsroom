
import sys, json, glob
sys.path.insert(0, '.')
from config.settings import RAW_DIR

files = sorted(glob.glob(str(RAW_DIR / '**' / '*_run.json'), recursive=True))
data  = json.load(open(files[-1]))
md    = data.get('market_data', [])

live  = [d for d in md if d.get('type') == 'market_data_live']
print('Live price records in last archive:', len(live))
for d in live:
    print(d)