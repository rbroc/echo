# Prompt Selection
PCA was run on low-level features `["doc_length", "n_tokens", "n_characters", "n_sentences"]` to get new PC components. See the `PCA` folder for those results.

To then investigate differences between model completions and human completions, euclidean distances were computed between the PC components of the human generation and each model generation (i.e., human-beluga, human-llama2_chat). See also [src/prompt_selection/distance.py](https://github.com/rbroc/echo/blob/main/src/prompt_selection/distance.py).

## Plotting
Interactive plots illustrate the distance scores and their corresponding completion by hovering over them:
1. [dailydialog](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/main/results/distance/all_PC_jitterplots/interactive/dailydialog.html)
2. [dailymail_cnn](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/main/results/distance/all_PC_jitterplots/interactive/dailymail_cnn.html)
3. [mrpc](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/main/results/distance/all_PC_jitterplots/interactive/mrpc.html)
4. [stories](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/main/results/distance/all_PC_jitterplots/interactive/stories.html)

Static plots can also be found in this [folder](https://github.com/rbroc/echo/tree/main/results/distance/all_PC_jitterplots/static).

## Medians
The medians of the distances were computed for each model, dataset and prompt number:

| dataset       | model       |   1.0 |   2.0 |   3.0 |   4.0 | 5.0   | 6.0   |
|---------------|-------------|-------|-------|-------|-------|-------|-------|
|               | beluga      |  0.39 |  0.52 |  0.40 |  0.27 |       |       |
| dailydialog   | llama2_chat |  1.16 |  1.38 |  1.42 |  1.07 |       |       |
|               | beluga      |  0.61 |  0.46 |  0.49 |  0.62 | 1.11  | 0.664 |
| dailymail_cnn | llama2_chat |  1.67 |  1.48 |  1.31 |  1.39 | 1.803 | 2.001 |
|               | beluga      |  0.04 |  0.05 |  0.05 |  0.05 |       |       |
| mrpc          | llama2_chat |  0.07 |  0.07 |  0.06 |  0.06 |       |       |
|               | beluga      |  2.83 |  2.98 |  5.05 |  3.14 | 3.533 | 2.875 |
| stories       | llama2_chat |  3.07 |  3.41 |  3.83 |  3.03 | 2.871 | 2.535 |

## Two lowest medians per MODEL, DATASET
We can also group the results in the two prompts that hold the lowest median per model and dataset: 
| dataset       | model       |   prompt |   median |
|---------------|-------------|----------|----------|
| dailydialog   | beluga      |        4 |    0.267 |
|               |             |        1 |    0.394 |
|               | llama2_chat |        4 |    1.074 |
|               |             |        1 |    1.162 |
| dailymail_cnn | beluga      |        2 |    0.46  |
|               |             |        3 |    0.487 |
|               | llama2_chat |        3 |    1.311 |
|               |             |        4 |    1.394 |
| mrpc          | beluga      |        1 |    0.045 |
|               |             |        3 |    0.046 |
|               | llama2_chat |        4 |    0.064 |
|               |             |        3 |    0.064 |
| stories       | beluga      |        1 |    2.832 |
|               |             |        6 |    2.875 |
|               | llama2_chat |        6 |    2.535 |
|               |             |        5 |    2.871 |