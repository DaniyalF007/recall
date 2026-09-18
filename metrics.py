import json, re

K_VALUES = [1, 3]
modes = ['Dense', 'BM25', 'Hybrid (RRF)']

def norm(s):
    s = s.lower().replace('-\n', '')     # rejoin hyphenated line breaks
    return re.sub(r'\s+', ' ', s)         # collapse newlines/spaces

results = json.load(open('data/results.json'))['results']
bench = {q['id']: q for q in json.load(open('data/benchmark.json'))['questions']}
gold = json.load(open('data/gold.json'))

def first_hit_rank(chunks, g):
    g = norm(g)
    for i, c in enumerate(chunks, start=1):
        if g in norm(c):
            return i
    return None

recall = {m: {k: 0 for k in K_VALUES} for m in modes}
rr = {m: 0.0 for m in modes}
never_found = []
n = 0
for r in results:
    if bench[r['id']]['type'] == 'negative':
        continue
    if r['id'] not in gold:
        continue
    n += 1
    hit_any = False
    for m in modes:
        rank = first_hit_rank(r['results'][m]['retrieved_chunks'], gold[r['id']])
        if rank:
            hit_any = True
            rr[m] += 1 / rank
            for k in K_VALUES:
                if rank <= k:
                    recall[m][k] += 1
    if not hit_any:
        never_found.append(r['id'])

print(f"\nPositive questions scored: {n}/40\n")
for m in modes:
    line = f"{m:14}"
    for k in K_VALUES:
        line += f"  R@{k} {recall[m][k]}/{n} ({100*recall[m][k]/n:.0f}%)"
    line += f"  MRR {rr[m]/n:.3f}"
    print(line)
if never_found:
    print("\nGold never matched (check wording OR genuinely never retrieved):")
    print(" ", ", ".join(never_found))