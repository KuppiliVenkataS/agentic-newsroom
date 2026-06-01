import sys, json
sys.path.insert(0, '.')
from graph.knowledge_graph import KnowledgeGraph
from prediction.predictor import generate_prediction
from report.generator import generate_report

kg         = KnowledgeGraph()
prediction = generate_prediction(kg)
path       = generate_report(prediction, kg)
print('Report saved to:', path)
print()
print(open(path).read())