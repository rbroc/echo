import ndjson
import json
from pathlib import Path


def cleanup():
    ''' Standardizes datasets by lowercasing and removing irregular format '''

    # Cleanup mrsp
    msrpath = Path('..') / 'data' / 'msrp' 
    msrfile = msrpath / 'raw.ndjson'
    with open(msrfile) as f:
        msrp = ndjson.load(f)
    for m in msrp:
        m['human_completions'] = m['human_completions'][0][0].lower()
        m['source'] = m['source'].lower()
    with open(msrpath / 'data.ndjson', 'w') as f:
        ndjson.dump(msrp, f)
    
    # Cleanup stories
    storiespath = Path('..') / 'data' / 'stories' 
    storiesfile = storiespath / 'raw.json'
    with open(storiesfile) as f:
        stories = json.load(f)
    for s in stories:
        s['source'] = s['source'].lower()
        s['human_completions'] = s['human_completions'].lower()
    with open(storiespath / 'data.ndjson', 'w') as f:
        ndjson.dump(stories, f)

    # Cleanup dailymail
    dmpath = Path('..') / 'data' / 'dailymail_cnn' 
    dmfile = dmpath / 'raw.ndjson'
    with open(dmfile) as f:
        dm = ndjson.load(f)
    for d in dm:
        d['human_completions'] = d['human_completions'][0][0].lower()
        d['source'] = d['source'].lower()
    with open(dmpath / 'data.ndjson', 'w') as f:
        ndjson.dump(dm, f)
    

if __name__ == '__main__':
    cleanup()