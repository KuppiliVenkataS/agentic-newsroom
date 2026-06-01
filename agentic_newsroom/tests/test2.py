import json
f = open('/Volumes/Mac_extension/projects/OilNewsDB/agentic_newsroom/data/processed/2026/05/2026-05-21_19-24-00_processed.json')
data = json.load(f)
# find first article with successful extraction
for a in data['articles']:
    ext = a.get('extraction', [])
    if ext and ext[0].get('status') == 'ok':
        print('TITLE:', a.get('title', a.get('gdelt_source', '')))
        print('EXTRACTION:')
        print(json.dumps(ext[0], indent=2))
        break