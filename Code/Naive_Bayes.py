from sklearn.pipeline import Pipeline
from Classification import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Naive Bayes
def NaiveBayes(train_data, train_labels, test_data, test_labels):
    print("Creating Naive Bayes Model")

    parameters = {
        'vect__max_features': [1500, 2000, 2500, 3000],
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'vect__max_df': [0.88, 0.9, 0.95, 0.97],
        'vect__min_df': [0, 0.01, 0.02],
        'nb__alpha': [0.5, 1]
    }

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('nb', MultinomialNB())
    ])

    build_pipeline_classification(pipeline, parameters, train_data, train_labels, test_data, test_labels)
