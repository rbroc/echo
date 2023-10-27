# Prompt Selection
### Euclidean Distances
To investigate differences between model completions and human completions, euclidean distances were computed between the PC components of the human generation and the model generation for each unique ID in each dataset (i.e., beluga-human, llama2_chat-human). 

See [src/prompt_selection/distance.py](src/prompt_selection/) for how these are computed. 

#### Plotting
The distances are plotted interactively in which you can hover over each point to view the completion: 
1. [dailydialog](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/prompt-select/results/distance/all_PC_jitterplots/interactive/dailydialog.html)
2. [dailymail_cnn](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/prompt-select/results/distance/all_PC_jitterplots/interactive/dailymail_cnn.html)
3. [mrpc](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/prompt-select/results/distance/all_PC_jitterplots/interactive/mrpc.html)
4. [stories](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/prompt-select/results/distance/all_PC_jitterplots/interactive/stories.html)