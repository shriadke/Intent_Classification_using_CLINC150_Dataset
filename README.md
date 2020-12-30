# Intent_Classification_using_CLINC150_Dataset

This project shows various machine learning methods that can be used for the classification of intents. Please make sure that libraries listed in [requirements](https://github.com/shriadke/Intent_Classification_using_CLINC150_Dataset/blob/main/requirements.txt) are installed within the working environment.

The data can be downloaded from [CLINC150](https://github.com/clinc/oos-eval) that contains 10 real-world problem domains having 150 distinct intent classes(with 15 classes in each domain), each with 100 train, 20 validation, 30 test samples. This project uses [data_full.json](https://github.com/shriadke/Intent_Classification_using_CLINC150_Dataset/blob/main/data_full.json) as input.

For this implementation, I have selected 2 distinct intents from each of the 10 domains resulting in 20 distinct intent-classes (The selection is explained in notebooks).

Notebook [Intent_classification](https://github.com/shriadke/Intent_Classification_using_CLINC150_Dataset/blob/main/Intent_classification.ipynb) demonstrates an implementation of 4 basic ML classifiers along with the training and evaluation of the data.

Notebook [Intent_classification_with_BiLSTM](https://github.com/shriadke/Intent_Classification_using_CLINC150_Dataset/blob/main/Intent_classification_with_BiLSTM.ipynb) shows the use of a simple Bidirectional LSTM classifier along with the training and evaluation of the data. This model can be further tuned and performance can be significantly improved. The [sample model](https://github.com/shriadke/Intent_Classification_using_CLINC150_Dataset/blob/main/bidirectionalModel.h5) is uploaded along with this repo.

Overall, The class “IntentClassifier” provided in [IntentClassifier.py](https://github.com/shriadke/Intent_Classification_using_CLINC150_Dataset/blob/main/IntentClassifier.py) can be initialized with the json file in given CLINC150 format. It takes several optional arguments such as “classes to consider”, “classifier models”, “train to be performed?”, “models to be evaluated?”. Each of which can be set at the time of instantiation.

To test the class run [IntentClassifier.py](https://github.com/shriadke/Intent_Classification_using_CLINC150_Dataset/blob/main/IntentClassifier.py) as: 

```
python3 IntentClassifier.py
```
