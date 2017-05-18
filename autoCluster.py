from sklearn.cluster import KMeans
import random
import numpy as np

maxClusters = 10
minClusters = 2
maxRange = 300
maxSigma = 10
maxPointsCluster = 100
minPointsCluster = 20
maxIter = 50
ll_a_min = []
ll_a_max = []

for k in range(maxIter):
    # generate clusters
    X = list()
    nClusters = random.randint(minClusters, maxClusters)
    for i in range(nClusters):
        nPoints = random.randint(minPointsCluster, maxPointsCluster)
        mu = random.randint(0, maxRange)
        sigma = maxSigma * random.random()
        X += [random.gauss(mu, sigma) for j in range(nPoints)]
    X = np.array(X).reshape(-1, 1)

    # apply kmeans
    llVar = list()
    for i in range(3):
        kmeans = KMeans(n_clusters=nClusters-1+i).fit(X)
        predict = kmeans.predict(X)
        var = sum((X[i] - kmeans.cluster_centers_[predict[i]])**2 for i in range(len(X))) / len(X)
        llVar.append(var+1)

    # calculate a_min and a_max
    a_min = llVar[1]/llVar[2]
    a_max = llVar[0]/llVar[1]
    ll_a_min.append(a_min)
    ll_a_max.append(a_max)
    print(a_max, a_min)

# take the 80% of the measurements
ll_a_max = sorted(ll_a_max)[int(maxIter*0.2):]
ll_a_min = sorted(ll_a_min)[:int(maxIter*0.8)]
print()
print(ll_a_max[0], ll_a_min[-1])
