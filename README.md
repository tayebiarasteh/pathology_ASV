# The effect of speech pathology on automatic speaker verification


Overview
------

* This is the official repository of the paper [**The effect of speech pathology on automatic speaker verification: a large-scale study**](https://www.nature.com/articles/s41598-023-47711-7).

Abstract
------
Our comprehensive assessments demonstrate that pathological speech overall faces heightened privacy breach risks compared to healthy speech.

### Prerequisites

The software is developed in **Python 3.9**. For the deep learning, the **PyTorch 1.13** framework is used.



Main Python modules required for the software can be installed from ./requirements:

```
$ conda env create -f requirements.yaml
$ conda activate pathology_ASV
```

**Note:** This might take a few minutes.


Code structure
---

Our source code for training and evaluation of the deep neural networks, speech analysis and preprocessing are available here.

1. Everything can be run from *./speaker_main.py*. 
* The data preprocessing parameters, directories, hyper-parameters, and model parameters can be modified from *./configs/config.yaml*.
* Also, you should first choose an `experiment` name (if you are starting a new experiment) for training, in which all the evaluation and loss value statistics, tensorboard events, and model & checkpoints will be stored. Furthermore, a `config.yaml` file will be created for each experiment storing all the information needed.
* For testing, just load the experiment which its model you need.

2. The rest of the files:
* *./data/* directory contains all the data preprocessing, and loading files.
* *./speaker_Train_Valid.py* contains the training and validation processes.
* *./speaker_Prediction.py* all the prediction and testing processes.



------
### In case you use this repository, please cite the original paper:

Tayebi Arasteh S, Weise T, Schuster M, et al. *The effect of speech pathology on automatic speaker verification: a large-scale study*. Scientific Reports (2023) 13:20476. 
https://doi.org/10.1038/s41598-023-47711-7

### BibTex

    @article {pathology_asv,
      author = {Tayebi Arasteh, Soroosh and Weise, Tobias, and Schuster, Maria and Noeth, Elmar and Maier, Andreas and Yang, Seung Hee},
      title = {The effect of speech pathology on automatic speaker verification: a large-scale study},
      year = {2023},
      pages = {20476},
      volume = {13},
      doi = {https://doi.org/10.1038/s41598-023-47711-7},
      journal = {Scientific Reports}
    }
