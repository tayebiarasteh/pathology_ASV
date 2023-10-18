# The effect of speech pathology on automatic speaker verification


Overview
------

* This is the official repository of the paper [**The effect of speech pathology on automatic speaker verification: A large-scale study**](https://arxiv.org/abs/2204.06450).
* Pre-print version: [https://arxiv.org/abs/2204.06450](https://arxiv.org/abs/2204.06450)

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

S. Tayebi Arasteh, T. Weise, M. Schuster, et al. *The effect of speech pathology on automatic speaker verification: A large-scale study*. arxiv.2204.06450, https://doi.org/10.48550/arXiv.2204.06450, 2022.

### BibTex

    @article {pathology_asv,
      author = {Tayebi Arasteh, Soroosh and Weise, Tobias, and Schuster, Maria and Noeth, Elmar and Maier, Andreas and Yang, Seung Hee},
      title = {The effect of speech pathology on automatic speaker verification: A large-scale study},
      year = {2022},
      doi = {10.48550/arXiv:2204.06450},
      journal = {arXiv}
    }
