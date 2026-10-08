"""Inspect PAGE line polygons and sample images without downloading entire ZIPs."""
import argparse
import io
import json
import random
import statistics
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
import numpy as np


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.session = requests.Session()
        self.session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                           status_forcelist=[429, 500, 502, 503, 504])))
        self.cache_start, self.cache = 0, b''

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        n = self.size-self.pos if n < 0 else min(n, self.size-self.pos)
        if n <= 0: return b''
        start = self.pos
        if not (self.cache_start <= start and start+n <= self.cache_start+len(self.cache)):
            end = min(self.size-1, start+max(n, 65536)-1)
            r = self.session.get(self.url, params={'range_start':start,'range_end':end},
                                 headers={'Range':f'bytes={start}-{end}'}, timeout=60)
            r.raise_for_status()
            if r.status_code != 206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                raise RuntimeError(f'Server did not honor range: {r.status_code} {r.headers}')
            self.cache_start, self.cache = start, r.content
        self.pos += n
        off = start-self.cache_start
        return self.cache[off:off+n]


def parse_xml(blob):
    root = ET.fromstring(blob)
    is_alto = root.tag.endswith('}alto')
    page = root.find('.//{*}Page')
    lines = root.findall('.//{*}TextLine')
    counts, polygons = [], []
    rectangles = 0
    for line in lines:
        coords = line.find('{*}Shape/{*}Polygon') if is_alto else line.find('{*}Coords')
        if is_alto and coords is not None:
            values = list(map(float, coords.get('POINTS','').split()))
            pts = list(zip(values[::2], values[1::2]))
        else:
            pts = [] if coords is None else [tuple(map(float,p.split(','))) for p in coords.get('points','').split()]
        unique = set(pts)
        counts.append(len(unique))
        if len(unique)>=3:
            polygons.append(pts)
            rectangles += len(set(x for x,y in unique))==2 and len(set(y for x,y in unique))==2
    attrs = None if page is None else dict(page.attrib)
    if is_alto and attrs is not None:
        attrs['imageWidth'], attrs['imageHeight'] = attrs.get('WIDTH'), attrs.get('HEIGHT')
        attrs['imageFilename'] = root.findtext('.//{*}fileName', '')
    return {'format':'ALTO' if is_alto else 'PAGE', 'page':attrs, 'lines':len(lines),
            'regions':len(root.findall('.//{*}TextRegion')), 'vertices':counts,
            'axis_aligned_rectangles':rectangles, 'polygons':polygons}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--record',required=True)
    ap.add_argument('--archive',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--samples',type=int,default=15)
    ap.add_argument('--image-archive')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    meta=requests.get(f'https://zenodo.org/api/records/{args.record}',timeout=30).json()
    (args.out/'record.json').write_text(json.dumps(meta,indent=2))
    files={f['key']:f for f in meta['files']}
    def archive(name):
        f=files[name]
        if f['size'] < 10000000:
            response=requests.get(f['links']['self'],timeout=60)
            response.raise_for_status()
            return zipfile.ZipFile(io.BytesIO(response.content))
        return zipfile.ZipFile(RangeFile(f['links']['self'],f['size']))
    z=archive(args.archive)
    names=z.namelist()
    (args.out/'archive_names.json').write_text(json.dumps(names,indent=2))
    xmls=[n for n in names if n.lower().endswith('.xml') and '__MACOSX' not in n]
    print(args.record,'entries',len(names),'XMLs',len(xmls),'folders',Counter('/'.join(n.split('/')[:-1]) for n in xmls).most_common(15),flush=True)
    for n in names:
        if n.lower().endswith(('.md','.csv','.txt')) and z.getinfo(n).file_size<20000:
            print('TEXT',n,z.read(n).decode('utf8',errors='replace')[:3500],flush=True)
            if len(names)>1000: break
    sampled=sorted(random.Random(42).sample(xmls,min(args.samples,len(xmls))))
    rows=[]
    for i,n in enumerate(sampled):
        blob=z.read(n)
        path=args.out/f'sample_{i:03d}.xml';path.write_bytes(blob)
        d=parse_xml(blob);d.pop('polygons');d['member']=n;d['local_xml']=str(path)
        rows.append(d)
        print('XML',n,'lines',d['lines'],'median vertices',statistics.median(d['vertices']) if d['vertices'] else None,flush=True)
    iz=archive(args.image_archive) if args.image_archive else z
    ims=[n for n in iz.namelist() if n.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff')) and '__MACOSX' not in n]
    image_checks=[]
    for row in rows:
        if len(image_checks)>=3:break
        if not row['page']:continue
        name=Path(row['page'].get('imageFilename','')).name
        stem=Path(row['member']).stem
        matches=[n for n in ims if Path(n).name==name or Path(n).stem==stem]
        if not matches:continue
        n=matches[0];blob=iz.read(n)
        path=args.out/f'image_{len(image_checks):02d}{Path(n).suffix}';path.write_bytes(blob)
        with Image.open(io.BytesIO(blob)) as im:
            arr=np.asarray(im.convert('RGB').resize((256,256)),dtype=float)
            check={'member':n,'local_image':str(path),'local_xml':row['local_xml'],
                   'mode':im.mode,'size':im.size,'xml_size':[row['page'].get('imageWidth'),row['page'].get('imageHeight')],
                   'mean_channel_difference':float(np.mean(abs(arr[:,:,0]-arr[:,:,1])+abs(arr[:,:,1]-arr[:,:,2])))}
        image_checks.append(check);print('IMAGE',check,flush=True)
    vertices=[v for row in rows for v in row['vertices']]
    report={'record':args.record,'archive':args.archive,'total_xml_files':len(xmls),'total_images':len(ims),
            'sample_pages':len(rows),'sample_lines':len(vertices),
            'median_vertices':statistics.median(vertices) if vertices else None,
            'vertex_histogram':dict(Counter(vertices)), 'sample_axis_aligned_rectangles':sum(r['axis_aligned_rectangles'] for r in rows),
            'pages':rows,'images':image_checks}
    (args.out/'audit.json').write_text(json.dumps(report,indent=2))
    print('DONE',args.out,flush=True)


if __name__=='__main__':main()
