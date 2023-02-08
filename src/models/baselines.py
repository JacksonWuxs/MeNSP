

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from nltk import word_tokenize, ngrams
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation


class TfidfSVMClassifier:
    def __init__(self, num_features=100, clf=None):
        self._stopwords = set(stopwords.words("english")) | set(punctuation)
        self._scaler = StandardScaler()
        self._stemmer = PorterStemmer()
        self._extractor = TfidfVectorizer(sublinear_tf=True, lowercase=False, analyzer=lambda x: x,
                                          max_features=num_features)
        if clf == "RF":
            self._classifier = RandomForestClassifier(n_estimators=50)
        if clf == "GB":
            self._classifier = GradientBoostingClassifier()
        if clf == "Vote":
            self._classifier = VotingClassifier(estimators=[
                 ("svm", SVC(gamma="auto", probability=True)),
                 ("lr", LogisticRegression(C=1000, penalty="l1", tol=0.005, solver="saga")),
                 ("nb", GaussianNB()),
                 ("tree", DecisionTreeClassifier()),
                 ("mlp", MLPClassifier(alpha=0.05))
                 ],
                 voting="hard")

    def fit(self, data):
        C = self._preprocess(data)
        X = self._extractor.fit_transform(C)
        X = self._scaler.fit_transform(X.todense())
        Y = data.get_labels()
        self._classifier.fit(X, Y)

    def predict(self, data):
        C = self._preprocess(data)
        X = self._extractor.transform(C)
        X = self._scaler.transform(X.todense())
        return self._classifier.predict(X)

    def _preprocess(self, data):
        C = map(str.lower, data.get_responses())
        C = map(lambda x: [self._stemmer.stem(_) for _ in word_tokenize(x)], C)
        C = map(lambda x: [_ for _ in x if _ not in self._stopwords], C)
        C = map(lambda x: ngrams(x, 2), C)
        return C
