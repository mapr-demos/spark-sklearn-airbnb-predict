import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# scikit-learn normal
# from sklearn.grid_search import GridSearchCV

# spark
from pyspark import SparkContext, SparkConf
from spark_sklearn import GridSearchCV

def top_tfidf_feats(row, features, top_n=25):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

# scikit-learn normal
# from sklearn.grid_search import GridSearchCV

# spark
#from pyspark import SparkContext, SparkConf
#from spark_sklearn import GridSearchCV

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from collections import Counter

LISTINGSFILE = '/mapr/tmclust1/user/mapr/pyspark-learn/airbnb/listings.csv'
REVIEWSFILE = '/mapr/tmclust1/user/mapr/pyspark-learn/airbnb/reviews.csv'

cols = ['id',
        'neighbourhood_cleansed',
        ]

rcols = [ 'listing_id', 'comments' ]

nbhs = [ 'Mission', 'South of Market', 'Western Addition' ]

# read the file into a dataframe
df = pd.read_csv(LISTINGSFILE, usecols=cols, index_col='id')
rdf = pd.read_csv(REVIEWSFILE, usecols=rcols)

# combine the reviews with the listings into a single dataframe
# indexed by listing ID
rdf = rdf.groupby(['listing_id'])['comments']. \
    apply(lambda x: ' '.join(x.astype(str))).reset_index()
rdf = rdf.set_index(rdf['listing_id'].astype(float))
df = pd.concat([df, rdf], axis = 1)

print "before filtering: %d" % len(df.index)
df = df.dropna(axis=0)
df = df[df.neighbourhood_cleansed.isin(nbhs)]
print "after filtering: %d" % len(df.index)

le = preprocessing.LabelEncoder().fit(df.neighbourhood_cleansed)
df['nbh'] = le.transform(df.neighbourhood_cleansed)

tfid = TfidfVectorizer()
ttext = tfid.fit_transform(df['comments'])

#print top_feats_in_doc(ttext, tfid.get_feature_names(), 1, 10)
#sys.exit(0)

print "%d %d" % (ttext.shape[0], len(df['nbh']))
X_train, X_test, y_train, y_test = \
    train_test_split(ttext, df['nbh'],
    test_size=0.2, random_state=1)

rs = 1
ests = [ neighbors.KNeighborsClassifier(3),
         RandomForestClassifier(random_state=rs) ]

ests_labels = np.array(['KNeighbors', 'RandomForest' ])

for i, e in enumerate(ests):
        e.fit(X_train, y_train)
        this_score = metrics.accuracy_score(y_test, e.predict(X_test))
        scorestr = "%s: Accuracy Score %0.2f" % (ests_labels[i],
                this_score)
        print
        print scorestr
        print "-" * len(scorestr)
        print metrics.classification_report(y_test,
                e.predict(X_test), target_names=le.classes_)

tuned_parameters = { "max_depth": [3, None],
               "max_features": [1, 'auto'],
               "min_samples_split": [1, 20],
               "n_estimators": [10, 300, 500] }
rf = RandomForestClassifier(random_state=rs)

# spark-sklearn
conf = SparkConf()
sc = SparkContext(conf=conf)
clf = GridSearchCV(sc, rf, cv=3,
       param_grid=tuned_parameters,
       scoring='accuracy')

# scikit-learn
# clf = GridSearchCV(rf, cv=2, scoring='accuracy',
#         param_grid=tuned_parameters,
#         verbose=True)

preds = clf.fit(X_train, y_train)
best = clf.best_estimator_
this_score = metrics.accuracy_score(y_test, best.predict(X_test))
scorestr = "RF / GridSearchCV: Accuracy Score %0.2f" % this_score
print
print scorestr
print "-" * len(scorestr)
print metrics.classification_report(y_test,
        best.predict(X_test), target_names=le.classes_)
