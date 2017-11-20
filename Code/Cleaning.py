import csv
from pprint import pprint
import pandas as pd
from sklearn import metrics
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOPWORDS = stopwords.words("english")

# Read csv file
posts = pd.read_csv(r'../Data/Data&Data Classification Challenge - Facebook - Training Set.csv', sep='\t')

# seperate data based on seller type
fake_seller = posts[posts['INDEX New'] == 'Fake Seller']
no_seller = posts[posts['INDEX New'] == 'No Seller']
reseller = posts[posts['INDEX New'] == 'Reseller']

# fraction of data considered for analysis
sample_ratio = 0.5
posts = pd.concat([fake_seller.sample(frac=0.4), reseller.sample(frac=0.4), no_seller.sample(frac=0.4)])


# Create a new column to store if the post is seller or not a seller
def check_seller(category):
    if category == 'No Seller':
        return False
    else:
        return True


# clean description
def description_cleaning(text):
    text = str(text).lower()
    text = re.sub('[^A-Za-z]', ' ', text)
    text = text.split()
    words = [w for w in text if not w in STOPWORDS]
    words = [PorterStemmer().stem(w) for w in text]
    words = ' '.join([w for w in words if len(w) >= 2])
    return words


def clean_labels(text):
    text = str(text).lower().split(',')
    text = ''.join(text)
    return text


def check_picture_label(label):
    if str(label) != 'nan' or len(label) == 0:
        return True
    return False


def check(count):
    if count > 0:
        return True
    return False


pd.set_option('display.max_colwidth', -1)

posts['Seller'] = posts['INDEX New'].apply(check_seller)
posts['has_likes'] = posts['nb_like'].apply(check)
posts['has_shares'] = posts['nb_share'].apply(check)
posts['picture_labels'] = posts['picture_labels'].apply(clean_labels)
posts['found_keywords'] = posts['found_keywords'].apply(clean_labels)
posts['has_picture_label'] = posts['picture_labels'].apply(check_picture_label)
# + ' ' + posts['found_keywords'] + posts['description'] + ' '
posts['description'] = posts['picture_labels'] + ' ' + posts['description']
posts['Clean Description'] = posts['description'].apply(description_cleaning)

# pprint(posts['Clean Description'][:10])
