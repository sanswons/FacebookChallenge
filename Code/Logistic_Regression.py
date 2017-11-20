from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from Classification import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def LogRegr(train_data, train_labels, test_data, test_labels):
    print("Creating Logistic Regression Model")

    parameters = {
        'vect__max_features': [1500, 2000, 2500, 3000],
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'vect__max_df': [0.88, 0.9, 0.95, 0.97],
        'vect__min_df': [0, 0.01, 0.02],
    }

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('logr', LogisticRegression())
    ])

    build_pipeline_classification(pipeline, parameters, train_data, train_labels, test_data, test_labels)
