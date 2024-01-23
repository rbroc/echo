## Overview 
`ai_datasets` contains all data that is artificially generated in all parts of the project. The folder is firstly seperated into two folders `HF` and `vLLM` for each implementation of the models.
```
├── HF
│   ├── FULL_DATA     <--- complete datasets
│   ├── SUBSET_DATA   <--- incomplete datasets
│   ├── prob_decoding <--- test folder w. a single Beluga7B dataset with temp=1, k=50 etc.
│   └── prompt_select <--- initial prompting w. greedy decoding used in phase 1 of the project (see #RoadMap on main README)
└── vLLM
    ├── FULL_DATA     <--- complete datasets 

```
`FULL_DATA` contains full length datasets whereas `SUBSET_DATA` is used for testing and therefore contains datasets of varying length (e.g., 1000 generations). Some initial phases of the project required playing with prompting. All files related to this phase are found in `prompt_select`.

Within each folder, folders denote all datasets generated with a particular dataset e.g.,: 
```
└── vLLM
    ├── FULL_DATA
    │   ├── beluga7b
    │   └── mistral7b
```