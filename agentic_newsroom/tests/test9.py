import sys
sys.path.insert(0, '.')
from graph.knowledge_graph import KnowledgeGraph
from prediction.predictor import generate_prediction
import json

kg = KnowledgeGraph()
pred = generate_prediction(kg)
print(json.dumps(pred, indent=2))