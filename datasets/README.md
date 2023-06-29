Data structure:
- Line-delimited json where each dataset is a dictionary: 
```
[{'id': '',
  'source': 'this is text to summarize',
  'human_completion': ['this is the first summary',
                       'this is the second summary']}]
```

Model-generated completions could later on be added to this (with 'model_completion' as key, and dictionary as value, where keys would be model names), or saved as a separate ndjson file, with same example id. 
It would look something like:
```
[{'id': '',
  'source': 'this is text to summarize',
  'model_completion': {'palm': 'this is how PaLM completes the sentence',
                       'alpaca': 'this is how Alpaca completes the sentence'}]}]
```
