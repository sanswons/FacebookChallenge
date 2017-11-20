from Feature_Extraction import *
from Cleaning import *
from Classification import *
from Naive_Bayes import *
from Logistic_Regression import *
from Ensemble import *
from SVM import *

if __name__ == '__main__':
    train = train_data['Clean Description'].tolist()
    test = test_data['Clean Description'].tolist()
    NaiveBayes(train, train_data['INDEX New'].tolist(), test, test_data['INDEX New'].tolist())
    LogRegr(train, train_data['INDEX New'].tolist(), test, test_data['INDEX New'].tolist())
    SVM(train, train_data['INDEX New'].tolist(), test, test_data['INDEX New'].tolist())
    Ensemble(train, train_data['INDEX New'].tolist(), test, test_data['INDEX New'].tolist())