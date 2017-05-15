import pandas as pd
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import silhouette_score
import multiprocessing
import time
import math
import operator
import numpy as np
import functools

# Functions to evaluate the performance of a split #

def gini(y, classes):
    ll = (y.count(c) / len(y) for c in classes)
    return sum(pr*(1-pr) for pr in ll)

def entropy(y, classes):
    ll = (y.count(c) / len(y) for c in classes)
    # vigilar probabilitat 0
    return sum(-pr*math.log2(pr) for pr in ll)

# Functions to evaluate the general performance of the predictive model #

def accuracy(pred, real):
    return sum(map(lambda x: x[0] == x[1], zip(pred, real))) / len(pred)

def precision(pred, real):
    return sum(map(lambda x: x[0] and x[1], zip(pred, real))) / sum(pred)

def recall(pred, real):
    return sum(map(lambda x: x[0] and x[1], zip(pred, real))) / sum(real)

def fScore(pred, real):
    return 2 * precision(pred, real) * recall(pred, real) / (precision(pred, real) + recall(pred, real))

# Functions used to do some calculations that cannot be done by a lambda function because the code is parallelized
# in some parts and it cannot acces to a local function as a lambda can be #

def decideCatAttr(x, atr):
    return x == atr

def decideNumAtrr(x, cl0, cl1, cl2):
    return (cl0 + cl1) / 2 <= x and x < (cl1 + cl2) / 2

def joinConditions(x, cond1, cond2):
    return cond1(x) or cond2(x)

def alwaysTrue(x):
    return True

def alwaysFalse(x):
    return False

# Functions that evaluate the performance of a kmeans clustering #

def perfKmeanVar(x, predict, kmeans, i):
    var = sum((x[i] - kmeans.cluster_centers_[predict[i]])**2 for i in range(len(x))) / len(x)
    return (var + 1) * 1.3**i # 1.3

def perfKmeansSilhouette(x, predict, kmeans=None, i=None):
    # the sample size should be not too large
    return -silhouette_score(x, predict, sample_size=5000) # valor petit -> millor

class DecisionTree:
    def __init__(self, X, y, classes, level=0, f=gini, condition=alwaysTrue, perfKmeans=perfKmeansSilhouette):
        self.attrSplit = None
        self.sons = []
        self.X = X
        self.y = y
        self.classNode = max([(self.y.count(i), i) for i in set(self.y)])[1]
        self.classes = classes
        self.level = level
        self.f = f
        self.condition = condition
        self.naiveBayes = GaussianNB().fit([elem[:3] for elem in X], y)
        self.perfKmeans = perfKmeans

    def autoSplit(self, minSetSize=50, giniReduction=0.01):
        """
        Splits recursively the tree
        """
        # print(self.level) # per debugar
        if len(self.X) > minSetSize:
            (gImp, idxAttr) = self.bestSplit()[0]
            if gImp + giniReduction < self.f(self.y, self.classes):
                self.splitNode(idxAttr)
                for son in self.sons:
                    son.autoSplit()

    def __generateSubsets(self, idxAttr):
        """
        :param idxAttr: Index of the attribute that will be used to split the dataset
        :return: A diccionary with key -> value of the attribute; value -> indexes of rows that have this attribute value and a function
        """
        if type(self.X[0][idxAttr]) == int or type(self.X[0][idxAttr]) == float:
            return self.__generateSubsetsNum(idxAttr)
        else:
            return self.__generateSubsetsCat(idxAttr)

    def __generateSubsetsCat(self, idxAttr):
        """
        :param idxAttr:
        :return:
        Splits the dataset using an attribute that has a categorical value
        """
        d = dict()
        for i in range(len(self.X)):
            if self.X[i][idxAttr] in d:
                d[self.X[i][idxAttr]][0].append(i)
            else:
                # d[self.X[i][idxAttr]] = ([i], lambda x, atr=self.X[i][idxAttr]: x == atr)
                d[self.X[i][idxAttr]] = ([i], functools.partial(decideCatAttr, atr=self.X[i][idxAttr]))
        return d

    def __generateSubsetsNum(self, idxAttr, i=0):
        """
        :param idxAttr:
        :return:
        Splits the dataset using an attribute that has a numerical value
        """
        x = [elem[idxAttr] for elem in self.X]
        maxClusters = len(set(x)) # there can't be more clusters than different values of the data
        x = np.array(x).reshape(-1,1)
        if i <= 1:
            kmeans = KMeans(n_clusters=1, n_jobs=1).fit(x)
            bestScore = math.inf
            i = 2
        else:
            if i > maxClusters:
                raise Exception("Too much number of clusters")
            bestScore = -1 # fa que a la primera volta del while entri dins de l'if i executi el break
            kmeans = KMeans(n_clusters=i, n_jobs=1).fit(x)

        while True:
            newKmeans = KMeans(n_clusters=i, n_jobs=1) # parallel kmeans, using all the processors // de moment no
            newPredict = newKmeans.fit_predict(x)
            newScore = self.perfKmeans(x, newPredict, newKmeans, i)
            if newScore >= bestScore or i > maxClusters:
                predict = kmeans.predict(x)
                break
            bestScore = newScore
            kmeans = newKmeans # copy?
            i += 1

        d = dict()
        clusters = [-math.inf] + sorted(kmeans.cluster_centers_.flatten().tolist()) + [math.inf]
        for i in range(1, len(clusters) - 1):
            d[i-1] = ([], functools.partial(decideNumAtrr, cl0=clusters[i-1], cl1=clusters[i], cl2=clusters[i+1]))
        # diccionary that translates the index that kmeans gives to a cluster to the index of this cluster ordered
        auxDict = dict((i, elem[0]) for (i, elem) in enumerate(sorted(enumerate(kmeans.cluster_centers_.flatten().tolist()), key=lambda x: x[1])))
        for (i, prt) in enumerate(predict):
            d[auxDict[prt]][0].append(i)

        # for e in d:
        #     print(d[e])
        return d

    def splitNode(self, idxAttr):
        """
        :param idxAttr: Index of the attribute used to split the node
        Splits the tree given an attribute
        """
        self.sons = list() # delete all previous sons
        d = self.__generateSubsets(idxAttr)
        for elem in sorted(d.keys()):
            newX = [self.X[i] for i in d[elem][0]]
            newY = [self.y[i] for i in d[elem][0]]
            self.sons.append(DecisionTree(newX, newY, self.classes, self.level + 1, self.f, d[elem][1]))
        self.attrSplit = idxAttr

    def _auxBestSplit(self, i):
        """
        :param i: I th attribute used to split the data
        :return: A tuple containing the gini impurity of that split and the index of the attribute used for this split
        This function is used to paralelize the function bestSplit
        """
        d = self.__generateSubsets(i) # no cal que en aquest cas __generateSubsets faci una copia de X
        gImp = 0
        for subs in d:
            newY = [self.y[i] for i in d[subs][0]]
            gImp += len(d[subs][0])/len(self.y) * self.f(newY, self.classes)
        return (gImp, i)

    def bestSplit(self):
        """
        :return: Calculate the split that has the lower gini impurity
        """
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        ll = pool.map(self._auxBestSplit, range(len(self.X[0])))
        return sorted(ll)

    def prune(self):
        """
        Eliminates all the sons of this node
        """
        self.sons = []

    def joinNodes(self, idxSons):
        """
        :param idxSons: List of indexes of the sons that will be joined
        Join into one node the sons specified by idxSons
        """
        idxSons = sorted(idxSons, reverse=True)
        newX = list()
        newY = list()
        # s'ha de mirar si funciona aquesta manera de fusionar les condicions de 2 nodes
        newCondition = alwaysFalse
        for i in idxSons:
            newX += self.sons[i].X
            newY += self.sons[i].y
            # lambda x: newCondition(x) or self.sons[i].condition(x)
            newCondition = functools.partial(joinConditions, cond1=newCondition, cond2=self.sons[i].condition)
            self.sons.pop(i)
        self.sons.append(DecisionTree(newX, newY, self.classes, self.level + 1, self.f, newCondition))

    def getSons(self):
        return self.sons

    def getNode(self, ll):
        """
        :param ll: List of node indexes
        :return: The node that is at the end of the path described in ll
        """
        if ll == []:
            return self
        if ll[0] < 0 or len(self.sons) <= ll[0]:
            raise Exception('First value of', ll, 'out of range')
        return self.sons[ll[0]].getNode(ll[1:])

    def _auxPredict(self, elem):
        currentNode = self
        t = True
        while t:
            t = False
            for son in currentNode.sons:
                if son.condition(elem[currentNode.attrSplit]):
                    currentNode = son
                    t = True
                    break
        return currentNode.classNode

    def predict(self, X):
        """
        :param X: [[attr1, attr2...], [attr1, attr2...]...]
        :return: The value y[i] has the prediction of X[i]
        """
        if not type(X[0]) == list:
            X = [X]
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # return [self._auxPredict(elem) for elem in X]
        return pool.map(self._auxPredict, X)

    def __str__(self):
        # La accuracy s'ha de generalitza per a datasets amb etiquetes diferents a True i False
        strTree = 'size: ' + str(len(self.y)) + '; Accuracy: ' + \
                  str(round(max([self.y.count(cl) for cl in self.classes]) / len(self.y), 4)) + \
                  '; Attr split: ' + str(self.attrSplit) + '; ' + self.f.__name__ + ': ' + \
                  str(round(self.f(self.y, self.classes), 4)) + "; Predict: " + str(self.classNode) + '\n'
                    # posar les funcions de accuracy i altres (recall, precision...) fora de la classe i
                    # cridar-les en aquest print
        for i in range(len(self.sons)):
            strTree += (self.level + 1) * '\t' + str(i) + ' -> ' + self.sons[i].__str__()
        return strTree

df = pd.read_csv('dadesSantPauProc.csv')
df2 = df.get(['diesIngr', 'nIngr', 'nUrg', 'estacioAny', 'diagPrinc']) # 'diagPrinc'
df3 = df.get(['reingres'])
aux2 = df2.values.tolist()
aux3 = df3.values.flatten().tolist()
dcTree = DecisionTree(aux2, aux3, [True, False], f=gini)
t = time.time()
dcTree.autoSplit(minSetSize=20, giniReduction=0.01)
# dcTree.splitNode(2)
# for son in dcTree.sons:
#     if gini(son.y, son.classes) > 0.38:
#         son.splitNode(2)
print(dcTree)
print(time.time() - t)
t = time.time()
exit(0)

while True:
    try:
        exec(input())
    except Exception as e:
        print(e)

ll = dcTree.predict(aux2)
print(fScore(ll, aux3))
print(time.time() - t)

# y = [0.5 < random.random() for i in range(5000000)]
# print(y.count(True))
# t = time.clock()
# print(gini(y, [True, False]), time.clock() - t)

# interactive
while True:
    try:
        eval(input())
    except:
        pass
