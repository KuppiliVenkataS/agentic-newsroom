import sys
sys.path.insert(0, '.')
import json, glob
from processing.cleaner import prepare_article

files = sorted(glob.glob('/Volumes/Mac_extension/projects/OilNewsDB/agentic_newsroom/data/raw/**/*_run.json', recursive=True))
data  = json.load(open(files[-1]))
gdelt = [a for a in data['articles'] if a.get('type') == 'gdelt_gkg'][0]
result = prepare_article(gdelt)
print('chunks:', result.get('chunks'))
print('chunk_count:', result.get('chunk_count'))