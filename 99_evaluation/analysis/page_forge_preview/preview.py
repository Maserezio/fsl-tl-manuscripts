import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import sys, json, random, time
from pathlib import Path
from collections import defaultdict
import numpy as np, cv2
sys.path.insert(0,'50_modelling/instance_segmentation/mask_rcnn')
from page_forge import PageForge
out=Path('99_evaluation/analysis/page_forge_preview')
for coll in sys.argv[1:]:
    root=Path('00_data/RQ3/matrix')/coll
    co=json.loads((root/'coco_instances/train.json').read_text())
    g=defaultdict(list)
    for a in co['annotations']: g[a['image_id']].append(a)
    recs=[(root/'images/train'/im['file_name'],im,g[im['id']]) for im in co['images']]
    t=time.time(); f=PageForge(recs); t1=time.time()-t
    rng=random.Random(0); n=[]
    for k in range(3):
        img,anns=f.sample(rng); n.append(len(anns))
        a=np.asarray(img).copy()
        for i,an in enumerate(anns):
            p=np.array(an['segmentation'][0]).reshape(-1,2).round().astype(np.int32)
            cv2.polylines(a,[p],True,[(220,30,30),(30,160,30),(30,60,220)][i%3],2)
        cv2.imwrite(str(out/f'{coll}_{k}.jpg'),cv2.resize(a,None,fx=0.5,fy=0.5)[:,:,::-1])
        if k==0: cv2.imwrite(str(out/f'{coll}_{k}_clean.jpg'),np.asarray(img)[:,:,::-1])
    print(coll,'lines',len(f.lines),'thick',round(f.median_thickness,1),'build',round(t1,1),'lines/page',n)
