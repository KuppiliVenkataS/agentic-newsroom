import json, glob
files = sorted(glob.glob('/Volumes/Mac_extension/projects/OilNewsDB/agentic_newsroom/data/raw/**/*_run.json', recursive=True))
data = json.load(open(files[-1]))
gdelt = [a for a in data['articles'] if a.get('type') == 'gdelt_gkg']
print('GDELT articles:', len(gdelt))
print('First GDELT chunks:', gdelt[0].get('chunks', 'NO CHUNKS KEY')[:2] if gdelt else 'none')
