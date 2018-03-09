import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB

train_all = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
subm = pd.read_csv('./input/sample_submission.csv')
#pseudo = pd.read_csv('./input/2_avg_fast_glove_svm_toxic.csv')

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

#print(len(pseudo))
#for col in label_cols:
#    pseudo.drop(pseudo[abs(pseudo[col]-0.5)<0.49].index, axis=0, inplace=True)
#print(len(pseudo))
#for col in label_cols:
#    pseudo[col] = [int(c) for c in pseudo[col]]

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy import sparse

class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        #y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(#solver = 'lbfgs', 
                                       C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self

COMMENT = 'comment_text'
train_all[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

import unidecode
train_all[COMMENT] = train_all[COMMENT].apply(lambda x: unidecode.unidecode(x.lower()))
test[COMMENT] = test[COMMENT].apply(lambda x: unidecode.unidecode(x.lower()))

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
skf = KFold(n_splits=5, random_state=42, shuffle=True)

index = -1
for train_index, valid_index in skf.split(range(len(train_all))):
    index+=1; print("Fold:", index)
    train = train_all.iloc[train_index,:]
    valid = train_all.iloc[valid_index,:]
    iid = valid["id"]
    
    y = train_all[label_cols].values
    y_train, y_valid = y[train_index,:], y[valid_index,:]
    #y_train, y_valid = np.vstack((y[train_index,:], pseudo[label_cols])), y[valid_index,:]
    
    n = train.shape[0]
    vec = TfidfVectorizer(ngram_range=(2,6),
                           #max_features=50000,
                           #vocabulary=s,
                           analyzer="char",
                           min_df=1, #1
                           max_df=0.9, #0.9
                           strip_accents='unicode', 
                           use_idf=1,
                           smooth_idf=1, 
                           sublinear_tf=1)
    train_x = vec.fit_transform(train[COMMENT])
    #train_x = vec.fit_transform(np.hstack((train[COMMENT],test[COMMENT][pseudo.index])))
    valid_x = vec.transform(valid[COMMENT])
    test_x = vec.transform(test[COMMENT])
    print(train_x.shape)
    print("Finish TF-DIF")
    
    preds = np.zeros((len(test), len(label_cols)))
    valid = np.zeros((len(valid), len(label_cols)))
    
    res = []
    for i, j in enumerate(label_cols):
        model = NbSvmClassifier(C=5)
        model.fit(train_x, y_train[:,i])
        valid[:,i] = model.predict_proba(valid_x)[:,1]
        r = roc_auc_score(y_valid[:,i], valid[:,i])
        res.append(r)
        print(" ",index,j,r)
        preds[:,i] = model.predict_proba(test_x)[:,1]
        
    print(index,np.mean(res))
    
    submid = pd.DataFrame({'id': subm["id"]})
    submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
    submission.to_csv('./output/nbsvm_'+str(index)+'_tst5.csv', index=False)
    
    submid = pd.DataFrame({'id': iid})
    submission = pd.concat([submid, pd.DataFrame(valid, columns = label_cols)], axis=1)
    submission.to_csv('./output/nbsvm_'+str(index)+'_val5.csv', index=False)