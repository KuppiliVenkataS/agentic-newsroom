
import sys
sys.path.insert(0, '.')
from vectordb.store import VectorStore

store = VectorStore()
results = store.search('OPEC supply cut impact on oil price', n_results=5)
for r in results:
    print(f'Score: {r["score"]:.3f} | Source: {r["metadata"]["source"]} | {r["metadata"]["title"][:80]}')
    print(f'  {r["chunk"][:120]}')
    print()