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
import pickle

# Functions to evaluate the performance of a split #

def gini(y, classes):
    """
    :param y: A list with the real classes of the data
    :param classes: A list containing all the different classes that appear in y
    :return: The gini impurity of y
    """
    ll = (y.count(c) / len(y) for c in classes)
    return sum(pr*(1-pr) for pr in ll)

def entropy(y, classes):
    """
    :param y: A list with the real classes of the data
    :param classes: A list containing all the different classes that appear in y
    :return: The entropy of y
    """
    ll = (y.count(c) / len(y) for c in classes)
    return sum(-pr*math.log2(pr+1E-60) for pr in ll)

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
    """
    :return: True if x and atr have the same value
    """
    return x == atr

def decideNumAtrr(x, cl0, cl1, cl2):
    """
    :param x: A number
    :param cl0: The center of a cluster
    :param cl1: The center of a cluster
    :param cl2: The center of a cluster
    :return: True if x is closer to cl1 than cl0 or cl2
    """
    return (cl0 + cl1) / 2 <= x and x < (cl1 + cl2) / 2

def joinConditions(x, cond1, cond2):
    """
    :param x: Any value
    :param cond1: A boolean function
    :param cond2: A boolean function
    :return: A function that returns true if cond1 or cond2 are true for the parameter x
    """
    return cond1(x) or cond2(x)

def alwaysTrue(x):
    """
    :return: True, always
    """
    return True

def alwaysFalse(x):
    """
    :return: False, always
    """
    return False

# Functions that evaluate the performance of a kmeans clustering #

def perfKmeanVar(x, predict, kmeans, nClusters):
    """
    :param x: List of numbers
    :param predict: A list of the cluster centers assigned to the numbers in x
    :param kmeans: A Kmeans object
    :param nClusters: Number of clusters
    :return: An indicator of how well the data is splited based on the variance and the number of clusters
    """
    var = sum((x[j] - kmeans.cluster_centers_[predict[j]])**2 for j in range(len(x))) / len(x)
    return (var + 1) * 1.3 ** nClusters # 1.3

def perfKmeansSilhouette(x, predict, kmeans=None, i=None):
    """
    :param x: List of numbers
    :param predict: A list of the cluster centers assigned to the numbers in x
    :return: The silhouette score
    """
    # the sample size should be not too large
    return -silhouette_score(x, predict, sample_size=1000) # valor petit -> millor

# Aux functions #

def delEmptyEntries(d):
    """
    :param d: A dictionary with value -> (list(), ...)
    Eliminates the entries that have as value[0] an empty list
    """
    for key in list(d.keys()):
        if len(d[key][0]) == 0:
            d.pop(key)

def automaticClustering(i, x, perfKmeans):
    """
    :param i: The number of clusters used to split the dataset
    :param x: The numerical data that have to be clustered
    :return: A kmeans object
    If i <= 1 the function tries to find the best number of clusters based on the function self.perfKmeans
    If i > 1 the function aplies kmeans with n_clusters=min(i, maxClusters)
    """
    maxClusters = len(set(x.flatten().tolist())) # there can't be more clusters than different values of the data
    if i > 1:
        i = min(i, maxClusters)
        kmeans = KMeans(n_clusters=i, n_jobs=1).fit(x)
    else:
        kmeans = KMeans(n_clusters=1, n_jobs=1).fit(x)
        newKmeans = kmeans
        bestScore = math.inf
        newScore = 999999
        i = 2

        while newScore < bestScore and i <= maxClusters:
            bestScore = newScore
            kmeans = newKmeans # copy?

            newKmeans = KMeans(n_clusters=i, n_jobs=1) # parallel kmeans, using all the processors // not now
            newPredict = newKmeans.fit_predict(x)
            newScore = perfKmeans(x, newPredict, newKmeans, i)
            i += 1
    return kmeans

# The main class #

class DecisionTree:
    def __init__(self, X, y, classes, attrNames=[], level=0, f=gini, condition=alwaysTrue, perfKmeans=perfKmeanVar, staticSplits=dict()):
        """
        :param X: Matrix with n rows, each row representing a sample, and m columns, each column representing the attributes of the sample
        :param y: Vector of n elements, each element i representing the value that corresponds to the ith entry in the matrix X
        :param classes: A list of the different values y can have
        :param level: Depth of the node (the root has level = 0)
        :param f: Function used to evaluate the performance of a split
        :param condition: The condition must accomplish the data from its parent to belong to this node
        :param perfKmeans: Function to evaluate the performance of the clustering of a numerical attribute using kmeans
        :param staticSplits: A dictionary with key -> index of an attribute; value -> how to split this attribute. For a numerical attribute the value is a list of numbers indicating the center of the clusters and for a categorical value is a list of lists, the second list containing the attributes that belong to a cluster
        """
        self.attrSplit = None
        self.sons = []
        self.X = X
        self.y = y
        self.classes = classes
        self.level = level
        self.f = f
        self.condition = condition
        self.perfKmeans = perfKmeans
        # staticSplits ha de ser coherent i amb el format correcte. No cal que sigui exhaustiu
        self.staticSplits = staticSplits#{4: [['_001', '_140', '_240', '_280', '_290', '_320', '_360', '_390', '_460', '_520', '_580', '_630'], ['_680', '_710', '_740', '_760', '_780', '_800']]}#{2: [1,3,5,7]}
        self.classNode = [(round(self.y.count(cls)/len(y), 3), cls) for cls in self.classes]
        if len(attrNames) != len(X[0]):
            self.attrNames = list(range(len(X[0])))
        else:
            self.attrNames = attrNames

        # Naive Bayes classifier (gaussian)
        self.gaussNB = GaussianNB()
        Xnum = [[atr for atr in elem if type(atr) == float or type(atr) == int] for elem in X]
        self.gaussNB.fit(Xnum, y)

        # Naive Bayes classifier (multinomial)
        Xcat = [[atr for atr in elem if type(atr) != float and type(atr) != int] for elem in X]
        self.typesCat = [set() for _ in range(len(Xcat[0]))]
        for row in Xcat:
            for i in range(len(row)):
                self.typesCat[i].add(row[i])
        self.typesCat = [list(s) for s in self.typesCat]

        Xmult = list()
        for row in Xcat:
            ll = self.__transformToBinary(row)
            Xmult.append(ll)
        self.multNB = MultinomialNB()
        self.multNB.fit(Xmult, y)

    def __transformToBinary(self, row):
        ll = []
        for i in range(len(row)):
            auxLL = [0] * len(self.typesCat[i])
            auxLL[self.typesCat[i].index(row[i])] = 1
            ll += auxLL
        return ll

    @staticmethod
    def copyVarTree(dcTree):
        """
        :param dcTree: A DecisionTree object
        :return: A new DecisionTree with the same values as dcTree
        This function is used when a DecisionTree is loaded from a file. The class may have been modified since the
        DecisionTree was stored. This helps to update the methods to the old DecisionTree
        """
        newDcTree = DecisionTree(X=dcTree.X, y=dcTree.y, classes=dcTree.classes, attrNames=dcTree.attrNames, level=dcTree.level, f=dcTree.f,
                                 condition=dcTree.condition, perfKmeans=dcTree.perfKmeans, staticSplits=dcTree.staticSplits)
        newDcTree.attrSplit = dcTree.attrSplit
        newDcTree.sons = [DecisionTree.copyVarTree(son) for son in dcTree.sons]
        return newDcTree

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
                    son.autoSplit(minSetSize, giniReduction)

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
        # create the same value -> ([], cond) for each value of the attribute idxAttr that must belong to the same node
        for st in self.staticSplits.get(idxAttr, []):
            cond = alwaysTrue
            for elem in st:
                auxCond = functools.partial(decideCatAttr, atr=elem)
                cond = functools.partial(joinConditions, cond1=cond, cond2=auxCond)
            value = ([], cond)
            for elem in st:
                d[elem] = value
        # put the index of the elements in X to the correct entry of the dictionary d
        for i in range(len(self.X)):
            if self.X[i][idxAttr] in d:
                d[self.X[i][idxAttr]][0].append(i)
            else:
                # d[self.X[i][idxAttr]] = ([i], lambda x, atr=self.X[i][idxAttr]: x == atr)
                d[self.X[i][idxAttr]] = ([i], functools.partial(decideCatAttr, atr=self.X[i][idxAttr]))
        # eliminate the elements in d that belong to the same split and eliminate the splits with 0 elements
        for st in self.staticSplits.get(idxAttr, []):
            for elem in st[1:]:
                d.pop(elem)
        delEmptyEntries(d)
        return d

    def __generateSubsetsNum(self, idxAttr, i=0):
        """
        :param idxAttr:
        :return:
        Splits the dataset using an attribute that has a numerical value
        """
        x = [elem[idxAttr] for elem in self.X]
        x = np.array(x).reshape(-1,1)
        if idxAttr in self.staticSplits:
            cls_centers = self.staticSplits[idxAttr]
            kmeans = KMeans(n_clusters=len(cls_centers))
            kmeans.cluster_centers_ = np.array(cls_centers).reshape(-1,1)
        else:
            kmeans = automaticClustering(i, x, self.perfKmeans)

        predict = kmeans.predict(x)
        d = dict()
        clusters = [-math.inf] + sorted(kmeans.cluster_centers_.flatten().tolist()) + [math.inf]
        for i in range(1, len(clusters) - 1):
            d[i-1] = ([], functools.partial(decideNumAtrr, cl0=clusters[i-1], cl1=clusters[i], cl2=clusters[i+1]))
        # diccionary that translates the index that kmeans gives to a cluster to the index of this cluster ordered
        auxDict = dict((i, elem[0]) for (i, elem) in enumerate(sorted(enumerate(kmeans.cluster_centers_.flatten().tolist()), key=lambda x: x[1])))
        for (i, prt) in enumerate(predict):
            d[auxDict[prt]][0].append(i)
        delEmptyEntries(d)
        return d

    def splitNode(self, idxAttr):
        """
        :param idxAttr: Index of the attribute used to split the node
        :return: The list of sons created
        Splits the tree given an attribute
        """
        self.sons = list() # delete all previous sons
        d = self.__generateSubsets(idxAttr)
        for elem in sorted(d.keys()):
            newX = [self.X[i] for i in d[elem][0]]
            newY = [self.y[i] for i in d[elem][0]]
            self.sons.append(DecisionTree(newX, newY, self.classes, self.attrNames, self.level + 1, self.f, d[elem][1], self.perfKmeans, self.staticSplits))
        self.attrSplit = idxAttr
        return self.sons

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
        ll = pool.map(self._auxBestSplit, range(len(self.X[0]))) # pool.map(self._auxBestSplit, range(len(self.X[0])))
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
        joinedNode = DecisionTree(newX, newY, self.classes, self.attrNames, self.level + 1, self.f, newCondition, self.perfKmeans, self.staticSplits)
        self.sons.append(joinedNode)
        return joinedNode

    def getSons(self):
        """
        :return: All the sons from the current node
        """
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

    def __bayesPredict(self, node, elem):
        """
        :param node: A DecisionTree
        :param elem: An element to predict
        :return: The prediction of elem based on the Naive Bayes classifier in node
        """
        numVar = list(filter(lambda x: type(x) == int or type(x) == float, elem))
        catVar = list(filter(lambda x: type(x) != int and type(x) != float, elem))
        catVarBinary = node.__transformToBinary(catVar)
        prGauss = node.gaussNB.predict_proba([numVar]).flatten()
        prMult = node.multNB.predict_proba([catVarBinary]).flatten()
        return [(prGauss[i] * prMult[i] / prCls, cls) for (i, (prCls, cls)) in enumerate(sorted(node.classNode, key=lambda x: x[1]))]

    def _auxPredict(self, elem, bayes):
        """
        :param elem: An element to predict
        :param bayes: If True it uses the Naive Bayes predictor
        :return: The prediction of elem
        """
        currentNode = self
        t = True
        while t:
            t = False
            for son in currentNode.sons:
                if son.condition(elem[currentNode.attrSplit]):
                    currentNode = son
                    t = True
                    break
        if bayes:
            return self.__bayesPredict(currentNode, elem)
        else:
            return currentNode.classNode

    def predict(self, X, bayes=False):
        """
        :param X: [[attr1, attr2...], [attr1, attr2...]...]
        :return: The value y[i] is a list of the probabilities that X[i] belongs to a certain class
        """
        if not type(X[0]) == list:
            X = [X]
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # return [self._auxPredict(elem) for elem in X]
        return list(map(functools.partial(self._auxPredict, bayes=bayes), X))

    def getNumElems(self):
        """
        :return: The number of elements that has the current node
        """
        return len(self.y)

    def getAccuracy(self):
        """
        :return: The most common class divided by the total number of elements
        """
        return round(max([self.y.count(cl) for cl in self.classes]) / len(self.y), 4)

    def getImpurity(self):
        """
        :return: The impurity of the current node
        """
        return round(self.f(self.y, self.classes), 4)

    def getPrediction(self):
        """
        :return: [(score1, class1), (score2, class2)...] sorted by score in decreasing order
        """
        return max(self.classNode)

    def getAttrSplit(self):
        """
        :return: The attribute used to split the data
        """
        if self.attrSplit == None:
            return None
        else:
            return self.attrNames[self.attrSplit]

    def getSegmentedData(self, attrId):
        """
        :param attrId: Index of an attribute
        :return: The data of the node segmented according to the attribute specified in attrId
        """
        M = [[] for _ in self.classes]
        for (i, elem) in enumerate(self.X):
            M[self.classes.index(self.y[i])].append(elem[attrId])
        return M

    def propagateChanges(self):
        """
        Propagate some parameters of the current node to all its sons recursively
        """
        for son in self.sons:
            son.f = self.f
            son.perfKmeans = self.perfKmeans
            son.staticSplits = self.staticSplits

    def __str__(self):
        # La accuracy s'ha de generalitza per a datasets amb etiquetes diferents a True i False
        strTree = 'size: ' + str(self.getNumElems()) + '; Accuracy: ' + \
                  str(self.getAccuracy()) + \
                  '; Attr split: ' + str(self.attrSplit) + '; ' + self.f.__name__ + ': ' + \
                  str(self.getImpurity()) + "; Predict: " + str(self.classNode) + '\n'
                    # posar les funcions de accuracy i altres (recall, precision...) fora de la classe i
                    # cridar-les en aquest print
        for i in range(len(self.sons)):
            strTree += (self.level + 1) * '\t' + str(i) + ' -> ' + self.sons[i].__str__()
        return strTree

if False:
    df = pd.read_csv('dadesSantPauProc.csv')
    df2 = df.get(['diesIngr', 'nIngr', 'nUrg', 'estacioAny', 'diagPrinc']) # 'diagPrinc'
    df3 = df.get(['reingres'])
    aux2 = df2.values.tolist()
    aux3 = df3.values.flatten().tolist()
    dcTree = DecisionTree(aux2, aux3, [True, False], f=gini)
    t = time.time()
    # dcTree.autoSplit(minSetSize=100, giniReduction=0.01)
    dcTree.splitNode(4)
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
