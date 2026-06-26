
from vectordb.store import VectorStore
store = VectorStore()
results = store.search('Hormuz vessel attack ship struck', n_results=10)
for r in results:
    print(r['metadata'].get('published',''), '|', r['metadata'].get('title','')[:70], '| score:', round(r['score'],3))