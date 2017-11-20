from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from Classification import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def Ensemble(train_data, train_labels, test_data, test_labels):
    print("Creating Random Forests Model")

    parameters = {
        'vect__max_features': [1500, 2000, 2500, 3000],
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'vect__max_df': [0.88, 0.9, 0.95, 0.97],
        'vect__min_df': [0, 0.01, 0.02],
        'rf__n_estimators': range(20, 81, 10)
    }

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('rf', RandomForestClassifier(min_samples_split=500, min_samples_leaf=50
                                      , max_depth=8, max_features='sqrt', random_state=10))
    ])

    build_pipeline_classification(pipeline, parameters, train_data, train_labels, test_data, test_labels)
