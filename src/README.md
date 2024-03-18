The src folder contains: 

| Folder         | Description                                                                                                   |
|----------------|---------------------------------------------------------------------------------------------------------------|
| `analysis`     | Scripts for analysis. Currently contains scripts to run PCA on low-level metrics, compute Euclidean distances, plot distances, check raw length distributions. |
| `clean`        | Clean and inspect human datasets.                                                                             |
| `generate`     | Entire generation pipeline here.                                                                               |
| `metrics`      | Extract metrics using test descriptives for human and AI data.                                                |
| `prompt_select`| Run the initial prompting pipeline + analysis of prompting.                                                    |
| `utils`        | Scripts for loading, preprocessing AI data (including combining with human datasets), running PCA, and computing distances.  |

Note that the `generate` folder has its own readme, detailing how to run the generation pipeline.