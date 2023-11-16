## Overview
This folder contains the following files ... 

## Generating Text 
To run the generation text pipeline, run in the terminal (from root):
```
bash generate.sh
```
Note that this will run text pipeline on a default dataset and model. See below for a custom pipeline.

### Custom Text Pipeline
To run a custom text generation pipeline, run in the terminal (from root): 
```
python src/generate/run_pipeline.py -mdl {MODELNAME} -f {FILENAME} -prompt_n {PROMPT_NUMBER} -subset {DATA_SUBSET}
```

With the arguments specified below: 
| <div style="width:80px">Arg</div>    | Description                             | <div style="width:120px">Default</div>    |
| :---        |:---                                                                                        |:---             |
|```-mdl```   | Model to use for generating. Choose between `beluga`, `falcon`, `falcon_instruct`, `llama2` and `llama2_chat`. Currently these are all the smaller 7B models. See full models names at [pipeline_fns.py](src/modules/pipeline_fns.py)            | `beluga`     |
|```-f```| Filename. Choose between `dailydialog`, `dailymail_cnn`, `mrpc` and `stories`  | `stories`.              |
|```-prompt_n```   | Integer between 1 and 6 currently. See [prompt_fns.py](src/modules/prompt_fns.py) for the specific prompts to choose between for each dataset.            |    `1`            |
|```-subset```   |   Integer how big of a subset of the data you want to use `dataset[:subset]``               | `None` (i.e., no subset)               |

*arguments subject to change*

Please note that these models are quite large and require 32 or 64 CPUs to run in decent time on a small subset. For running on CPU, you may want to avoid the `stories` dataset as it is a quite heavy!