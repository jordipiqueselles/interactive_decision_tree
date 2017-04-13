import pandas as pd
from sklearn.cluster import KMeans
import multiprocessing
import time
import math
import statistics
import numpy as np

def gini(y, classes):
    ll = (y.count(c) / len(y) for c in classes)
    return sum(pr*(1-pr) for pr in ll)

def entropy(y, classes):
    ll = (y.count(c) / len(y) for c in classes)
    return sum(-pr*math.log2(pr) for pr in ll)

class DecisionTree:
    def __init__(self, X, y, classes, level=0, f=gini, condition=lambda x: True):
        self.attrSplit = None
        self.sons = []
        self.X = X
        self.y = y
        self.classes = classes
        self.level = level
        self.f = f
        self.condition = condition

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
        :return: A diccionary: key -> one of the values of the selected attribute; value -> (Xi,yi) that have this attribute value
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
                d[self.X[i][idxAttr]][0].append(self.X[i])
                d[self.X[i][idxAttr]][1].append(self.y[i])
            else:
                d[self.X[i][idxAttr]] = ([self.X[i]], [self.y[i]], lambda x, atr=self.X[i][idxAttr]: x == atr)
        return d

    def __generateSubsetsNum(self, idxAttr, i=0):
        """
        :param idxAttr:
        :return:
        Splits the dataset using an attribute that has a numerical value
        """
        x = [elem[idxAttr] for elem in self.X]
        if i <= 0:
            i = 2
            bestScore = (statistics.variance(x) + 1)*3
        else:
            bestScore = -1
        x = np.array(x).reshape(-1,1)

        while True:
            kmeans = KMeans(n_clusters=i)
            predict = kmeans.fit_predict(x)
            var = sum((x[i] - kmeans.cluster_centers_[predict[i]])**2 for i in range(len(x))) / len(x)
            if (var+1) * 2**i >= bestScore:
                # hauria d'agafar el nClusters que dona el millor bestScore, no el nClusters+1 després del bestScore
                break
            bestScore = (var+1) * 2 ** i
            i += 1

        d = dict()
        clusters = [-math.inf] + sorted(kmeans.cluster_centers_.flatten().tolist()) + [math.inf]
        for i in range(1, len(clusters) - 1):
            d[i-1] = ([], [], lambda x, cl0=clusters[i-1], cl1=clusters[i], cl2=clusters[i+1]: \
                (cl0 + cl1) / 2 <= x and x < (cl1 + cl2) / 2)
        # diccionary that translates the index that kmeans gives to a cluster to the index of this cluster ordered
        auxDict = dict((i, elem[0]) for (i,elem) in enumerate(sorted(enumerate(kmeans.cluster_centers_.flatten().tolist()), key=lambda x: x[1])))
        for (i, prt) in enumerate(predict):
            d[auxDict[prt]][0].append(self.X[i])
            d[auxDict[prt]][1].append(self.y[i])
        return d

    def splitNode(self, idxAttr):
        """
        :param idxAttr: Index of the attribute used to split the node
        Splits the tree given an attribute
        """
        self.sons = list() # delete all previous sons
        d = self.__generateSubsets(idxAttr)
        for elem in sorted(d.keys()):
            self.sons.append(DecisionTree(d[elem][0], d[elem][1], self.classes, self.level + 1, self.f, d[elem][2]))
        self.attrSplit = idxAttr

    def _auxBestSplit(self, i):
        """
        :param i: I th attribute used to split the data
        :return: A tuple containing the gini impurity of that split and the index of the attribute used for this split
        This function is used to paralelize the function bestSlpit
        """
        d = self.__generateSubsets(i)
        gImp = sum((len(d[subs][1])/len(self.y))*self.f(d[subs][1], self.classes) for subs in d)
        return (gImp, i)

    def bestSplit(self):
        """
        :return: Calculate the split that has the lower gini impurity
        """
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        ll = pool.map(self._auxBestSplit, range(len(self.X[0])))
        return sorted(ll)

    def prune(self, idxSons=[]):
        """
        :param idxSons: List of indexes of the sons that will be pruned
        Eliminates the sons described in idxSons and restores its data back to the parent node
        """
        if idxSons == []:
            idxSons = list(range(len(self.sons)))
        idxSons = sorted(idxSons, reverse=True)
        for i in idxSons:
            self.sons.pop(i)

    def joinNodes(self, idxSons):
        """
        :param idxSons: List of indexes of the sons that will be joined
        Join into one node the sons specified by idxSons
        """
        idxSons = sorted(idxSons, reverse=True)
        newX = list()
        newY = list()
        # s'ha de mirar si funciona aquesta manera de fusionar les condicions de 2 nodes
        newCondition = lambda x: False
        for i in idxSons:
            newX += self.sons[i].X
            newY += self.sons[i].y
            newCondition = lambda x: newCondition(x) or self.sons[i].condition(x)
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

    def predict(self, X):
        """
        :param X: [[attr1, attr2...], [attr1, attr2...]...]
        :return: The value y[i] has the prediction of X[i]
        """
        if not type(X[0]) == list:
            X = [X]
        y = list()
        for elem in X:
            currentNode = self
            t = True
            while t:
                t = False
                for son in currentNode.sons:
                    if son.condition(elem[currentNode.attrSplit]):
                        currentNode = son
                        t = True
                        break
            m = max([(currentNode.y.count(i), i) for i in set(currentNode.y)])
            y.append(m[1])
        return y

    def __str__(self):
        # La accuracy s'ha de generalitza per a datasets amb etiquetes diferents a True i False
        strTree = 'size: ' + str(len(self.y)) + '; Accuracy: ' + str(max(self.y.count(True), self.y.count(False)) / len(self.y)) + \
                  '; Attr split: ' + str(self.attrSplit) + '; ' + self.f.__name__ + ': ' + str(self.f(self.y, self.classes)) + '\n'
        for i in range(len(self.sons)):
            strTree += (self.level + 1) * '\t' + str(i) + ' -> ' + self.sons[i].__str__()
        return strTree

df = pd.read_csv('dadesSantPauProc.csv')
df2 = df.get(['diesIngr', 'nIngr', 'nUrg', 'estacioAny', 'diagPrinc']) # 'diagPrinc'
df3 = df.get(['reingres'])
aux2 = df2.values.tolist()
aux3 = df3.values.flatten().tolist()
dcTree = DecisionTree(aux2, aux3, [True, False], f=gini)
t = time.clock()
#dcTree.autoSplit(minSetSize=20, giniReduction=0.01)
dcTree.splitNode(2)
# for son in dcTree.sons:
#     if gini(son.y, son.classes) > 0.38:
#         son.splitNode(2)
print(dcTree)
ll = dcTree.predict(aux2)
print(ll.count(True), ll.count(False))
# print(time.clock() - t)
pass
# y = [0.5 < random.random() for i in range(5000000)]
# print(y.count(True))
# t = time.clock()
# print(gini(y, [True, False]), time.clock() - t)