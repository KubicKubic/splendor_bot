"""Fetch the public, versioned HullQin bundle and extract its factual card table.

No JavaScript is executed by this extractor. Snapshots permit offline audits.
"""
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://fe-1255520126.file.myqcloud.com/game/static/js/ccbs.73a7734d.chunk.js'


def extract(source):
    text = source.split('3797:function(e,n,t){', 1)[1]
    def array(pattern):
        return json.loads(re.search(pattern, text).group(1))
    nobles = array(r'a=(\[\[.*?\]\]),c=')
    points = array(r',c=(\[[\d,]+\]);') * 5
    tier = array(r',o=(\[[\d,]+\]);') * 5
    base = array(r'var l=(\[\[.*?\]\]),s=')
    costs = []
    for color in range(5):
        costs.extend([row[-color:] + row[:-color] if color else row[:] for row in base])
    for index, values in re.findall(r's\[(\d+)\]=(\[[\d,]+\])', text):
        costs[int(index)] = json.loads(values)
    cards = [dict(id=i, tier=tier[i], bonus=i // 18, points=points[i], cost=costs[i]) for i in range(90)]
    assert [sum(c['tier'] == t for c in cards) for t in range(3)] == [40, 30, 20]
    return dict(colors=['white', 'blue', 'green', 'red', 'black'], cards=cards, nobles=nobles)


if __name__ == '__main__':
    source = urlopen(URL, timeout=30).read()
    target = ROOT / 'reference'
    target.mkdir(parents=True, exist_ok=True)
    (target / 'ccbs.73a7734d.chunk.js').write_bytes(source)
    data = extract(source.decode())
    data['source'] = dict(url=URL, sha256=hashlib.sha256(source).hexdigest(), retrieved='2026-09-09')
    (ROOT / 'data').mkdir(exist_ok=True)
    (ROOT / 'data' / 'cards.json').write_text(json.dumps(data, indent=2) + '\n')
    print(data['source'])
