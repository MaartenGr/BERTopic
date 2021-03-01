from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.utils import check_array
import numpy as np
import scipy.sparse as sp


class ClassTFIDF(TfidfTransformer):
    """
    A Class-based TF-IDF procedure using scikit-learns TfidfTransformer as a base.

    ![](../img/ctfidf.png)

    C-TF-IDF can best be explained as a TF-IDF formula adopted for multiple classes
    by joining all documents per class. Thus, each class is converted to a single document
    instead of set of documents. Then, the frequency of words **t** are extracted for
    each class **i** and divided by the total number of words **w**.
    Next, the total, unjoined, number of documents across all classes **m** is divided by the total
    sum of word **i** across all classes.
    """
    def __init__(self, *args, **kwargs):
        super(ClassTFIDF, self).__init__(*args, **kwargs)
        self.df = None

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights).

        Arguments:
            X: A matrix of term/token counts.
            n_samples: Number of total documents
        """
        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = np.float64

        if self.use_idf:
            _, n_features = X.shape
            self.df = np.squeeze(np.asarray(X.sum(axis=0)))
            avg_nr_samples = int(X.sum(axis=1).mean())
            idf = np.log(avg_nr_samples / self.df)
            self._idf_diag = sp.diags(idf, offsets=0,
                                      shape=(n_features, n_features),
                                      format='csr',
                                      dtype=dtype)

        return self

    def transform(self, X: sp.csr_matrix, copy=True):
        """Transform a count-based matrix to c-TF-IDF

        Arguments:
            X (sparse matrix): A matrix of term/token counts.

        Returns:
            X (sparse matrix): A c-TF-IDF matrix
        """
        if self.use_idf:
            X = normalize(X, axis=1, norm='l1', copy=False)
            X = X * self._idf_diag

        return X
