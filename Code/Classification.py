from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from Feature_Extraction import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def build_pipeline_classification(pipeline, parameters, train_data, train_labels, test_data, test_labels):
    print('Grid Search')
    grid_search = GridSearchCV(pipeline, parameters)
    classifier = grid_search.fit(train_data, train_labels)
    print('Found best parameters')
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)
    generate_results(test_data, test_labels, classifier)


def generate_results(test_matrix, test_labels, model):
    prediction = model.predict(test_matrix)
    print(confusion_matrix(test_labels, prediction))
    print(accuracy_score(test_labels, prediction))
    print(classification_report(test_labels, prediction))
