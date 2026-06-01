import sys
sys.path.insert(0, '.')
from graph.knowledge_graph import KnowledgeGraph
kg = KnowledgeGraph()
print('Signal summary:', kg.query_signal_summary())
print('Top orgs:', kg.query_top_organisations())
print('Recent events:', kg.query_recent_events(5))