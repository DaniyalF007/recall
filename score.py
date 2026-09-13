import json

# Human-verified labels against ground truth.
# Positives: 1 if answer conveys the ground-truth fact. Negatives: 1 if correctly refused.
# (type, Dense, BM25, Hybrid)   type: f=factual p=paraphrased i=inferential n=negative
labels = {
1:('f',1,0,1),2:('f',0,1,1),3:('f',0,1,0),4:('f',1,1,1),5:('p',1,0,1),
6:('p',1,1,1),7:('p',1,0,0),8:('i',0,1,1),9:('i',0,1,0),10:('n',1,1,1),
11:('n',1,1,0),12:('n',1,1,1),13:('f',1,0,0),14:('f',0,1,0),15:('p',0,0,1),
16:('n',1,1,1),17:('i',0,0,0),18:('p',1,0,1),19:('f',0,1,0),20:('n',1,1,1),
21:('f',0,1,1),22:('f',0,1,0),23:('f',1,0,1),24:('p',1,0,1),25:('p',0,1,0),
26:('f',0,0,0),27:('p',0,0,0),28:('i',0,1,0),29:('i',0,0,1),30:('n',1,1,1),
31:('n',1,1,1),32:('f',0,1,0),33:('p',1,1,1),34:('i',0,0,0),35:('n',1,1,1),
36:('f',1,0,0),37:('f',1,0,1),38:('f',1,1,1),39:('f',0,0,1),40:('f',1,0,1),
41:('p',1,1,1),42:('p',1,1,1),43:('f',1,1,1),44:('f',0,1,1),45:('f',1,1,1),
46:('p',1,1,1),47:('i',0,1,1),48:('i',1,1,1),49:('n',1,1,1),50:('n',1,1,1),
}

modes = ['Dense', 'BM25', 'Hybrid']
groups = {'Overall': lambda t: True, 'Factual': lambda t: t == 'f',
          'Paraphrased': lambda t: t == 'p', 'Inferential': lambda t: t == 'i',
          'Negative': lambda t: t == 'n'}

for name, sel in groups.items():
    row = []
    for i, m in enumerate(modes):
        picked = [v[1 + i] for v in labels.values() if sel(v[0])]
        row.append(f"{m} {sum(picked)}/{len(picked)} ({100*sum(picked)/len(picked):.0f}%)")
    print(f"{name:12} " + "  ".join(row))
