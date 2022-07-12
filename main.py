import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_diabetes, make_regression, load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score, train_test_split
from statistics import mean
from operator import itemgetter


def score(harmony):
    # selected_xtrain = np.ndarray.copy(X_train)
    # selected_xtest = np.ndarray.copy(X_test)
    selected_x = np.ndarray.copy(X)
    deleteindex = []

    for idx, val in enumerate(harmony):
        if val == False:
            deleteindex.append(idx)
    selected_x = np.delete(selected_x, deleteindex, 1)
    return f1_score(selected_x)


def decision_tree_score(selected_x):
    clf = DecisionTreeClassifier()
    return 1 - (mean(cross_val_score(clf, selected_x, y, cv=5)))


def regression_score(selected_x):
    regr = linear_model.LinearRegression()
    return mean(cross_val_score(regr, selected_x, y, cv=5,scoring="neg_mean_squared_error"))*-1

def f1_score(selected_x):
    clf = GaussianNB()
    return 1 - (mean(cross_val_score(clf, selected_x, y, cv=5,scoring = "f1")))

def load_dataset_classification():
    data = load_iris()
    X, y = data["data"], data["target"]
    noise = np.random.uniform(0, 10, size=(X.shape[0], 5))
    X = np.hstack((X, noise))
    return X, y

def load_dataset_breast_cancer():
    data = load_breast_cancer()
    X, y = data["data"], data["target"]
    return X, y

def load_dataset_regression():
    data = load_diabetes()
    X, y = data["data"], data["target"]
    noise = np.random.uniform(0, 10, size=(X.shape[0], 5))
    X = np.hstack((X, noise))
    return X, y


def create_dataset_regression():
    Xcreate, ycreate= make_regression(
        n_samples=1000,
        n_features=100,
        n_informative=10,
        noise=8,
    )
    return Xcreate,ycreate

np.random.seed(0)
X, y = load_dataset_breast_cancer()

hmcr = .9
par = .3
dimension = len(X[0])
hms = 10
iterasi = 1000
bestscore = []
for iteration in range(10):
    print(iteration)
    hm = []
    for i in range(hms):
        harmony = np.random.randint(2, size=dimension)
        skor = score(harmony)
        hm.append({"harmony": harmony, "skor": skor})
    hm = sorted(hm, key=itemgetter('skor'))

    for i in range(iterasi):
        harmonybaru = [1] * dimension
        for j in range(dimension):
            if hmcr >= random.random():
                harmonybaru[j] = random.choice(hm)["harmony"][j]
                if par >= random.random():
                    if harmonybaru[j] == 0:
                        harmonybaru[j] = 1
                    else:
                        harmonybaru[j] = 0
            else:
                harmonybaru[j] = np.random.randint(2)
        skorbaru = score(harmonybaru)
        if skorbaru < hm[hms - 1]["skor"]:
            hm[hms - 1]["harmony"] = harmonybaru
            hm[hms - 1]["skor"] = skorbaru
            hm = sorted(hm, key=itemgetter('skor'))
    bestscore.append(hm[0]["skor"])
# plt.plot(bestscore)
# plt.ylabel('Score')
# plt.xlabel('Iteration')
# plt.title('HS for feature selection')
# plt.show()
print(bestscore)
print("min = ", min(bestscore))
print("mean = ", mean(bestscore))
print("max = ", max(bestscore))
print("best")
print("Score = ", hm[0]["skor"])
print("Harmony = ", hm[0]["harmony"])
print("best")
print("Score = ", score([1,1,1,1,0,0,0,0,0]))
print("Harmony = ", [1,1,1,1,0,0,0,0,0])
