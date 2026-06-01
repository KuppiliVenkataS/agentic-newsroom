import sys, json, glob
sys.path.insert(0, '.')
from graph.knowledge_graph import KnowledgeGraph

# Load the processed file that has extraction results
files = sorted(glob.glob('/Volumes/Mac_extension/projects/OilNewsDB/agentic_newsroom/data/processed/**/*_processed.json', recursive=True))
print('Processed files found:', len(files))
if files:
    data = json.load(open(files[0]))
    articles = data.get('articles', [])
    has_extraction = [a for a in articles if a.get('extraction')]
    print('Articles with extraction:', len(has_extraction))
    
    kg = KnowledgeGraph()
    added = kg.add_articles(articles)
    print('Added to graph:', added)
    print('Signal summary:', kg.query_signal_summary())
    print('Top orgs:', kg.query_top_organisations())