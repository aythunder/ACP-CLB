# ACP-CLB
In this study, we propose a novel deep learning framework, ACP-CLB, for the identification of ACPs, based on the integration of multi-channel features. Specifically, it employs CNN to process FEGS features, BiLSTM to handle evolutionary information and physicochemical properties, and the large language model ProtBERT, with a three-channel fusion approach.
# How do we use protbert?
## 1. Environment setup
* Firstly, you need to create a virtual environment and set python==3.8 .
  ```pyhton
  conda create --name yourEnv python=3.8
  conda activate yourEnv
  ```
* Then, if you want to run this model, please install package in requirements.txt
  ```pyhton
  pip install -r requirements.txt 
  ```
* Finally, we utilize the ProtBERT model proposed by Elnaggar et al. in 2021. <br>
  The pre-trained parameters can be downloaded from [ProtBERT](https://huggingface.co/Rostlab/prot_bert). <br>
  Download the model parameter file and save it to a folder. Remember to change the path to the model.py file.
## 2. Data preparation
* The ACP740 dataset is stored in the data folder, and the ACPmain dataset is stored in the ACP-main folder.
* Using the FEGS.py file, extract the FEGS feature.<br>( We've already put the feature files in FEGS_dataset.pt .)
* Using the BLOUSE_AAindex.py file, extract the AAindex feature and BLOSUM62 feature.<br>( We've already put the feature files in LSTM_dataset.pt .)
## 3. Train model and test.
* The code for training and testing is in train.py
# Contact
Please feel free to submit any inquiries, issues, or bug reports.
