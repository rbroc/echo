### Encodings for additional information
The dataset contains a number of manual annotations, which could be useful throughout the analysis:
- act: a list of classification labels, with possible values including __dummy__ (0), inform (1), question (2), directive (3) and commissive (4).
- emotion: a list of classification labels, with possible values including no emotion (0), anger (1), disgust (2), fear (3), happiness (4), sadness (5) and surprise (6).

## Note on `raw.ndjson`
Please note that the `raw.ndjson` has already been processed slightly in `loader.py` but is named as such to match the file structure of the other datasets. The file `data.ndjson` thus refers to the standardized version (lowercasing etc.) that is performed on all datasets. 