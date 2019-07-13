'''
*   Created On: 12 July 2019
*   Status: Under Development
*   Author: Joshua Jansen Van Vuren
*   Description: Simple k-Means implementation
*   Clustering of N-data points with d dimensions with k clusters
'''

'''
*   Data should be of the format np.array(nDataPoints,nDimensions)
'''

#Imports
import numpy as np
import random
import math

#numClusters_k = 0
#numDataPoints_N = 0
#numDimensions_d = 0

class kMeans:
    def __init__(self, desiredClusters, steps, data):
        self.numDesiredClusters = desiredClusters
        self.data = data
        self.dataClusterLabels = np.zeros([data.shape[0]])
        self.clusterMeans = np.zeros([self.numDesiredClusters,data.shape[1]])
        self.dataPoints = data.shape[0]
        self.dataDimensions = data.shape[1]
        self.steps = steps

    def initLabels(self,initType):
        #TODO: switch case with different init types

        #Random init
        random.seed(5)
        for i in np.arange(self.numDesiredClusters):
            #For each cluster randomly pick a mean
            for j in np.arange(self.dataDimensions):
                self.clusterMeans[i,j] = random.randint(-10,10)

        self.updateClusterLabels()

        return self.clusterMeans

    def updateMeans(self):
        #Loop through each cluster
        for i in np.arange(self.numDesiredClusters):
            #Loop through each data point
            temp_sum = np.zeros(self.dataDimensions)
            temp_cluster_count = 0

            for j in np.arange(self.dataPoints):
                #If data point belongs to cluster add it to mean calculation
                if self.dataClusterLabels[j] == i:
                    temp_sum += self.data[j,:]
                    temp_cluster_count += 1

            if temp_cluster_count != 0:
                self.clusterMeans[i,:] = temp_sum/temp_cluster_count
            else:
                self.clusterMeans[i,:] = temp_sum

    def updateClusterLabels(self):
        #Assign each point to the nearest cluster mean
        for i in np.arange(self.dataPoints):
            dist = self.calculateDistance(self.data[i,:],self.clusterMeans[0,:])
            self.dataClusterLabels[i] = 0
            for j in np.arange(self.numDesiredClusters):
                newDist = self.calculateDistance(self.data[i,:],self.clusterMeans[j,:])
                if dist > newDist:
                    dist = newDist
                    self.dataClusterLabels[i] = j

    def converge(self):
        #Perform essentially the EM algorithm for k-Means
        for i in np.arange(self.steps):
            self.updateClusterLabels()
            self.updateMeans()

##########################All#The#Accessor#Methods#############################

    def getClusterLabels(self):
        return self.dataClusterLabels

    def getCluterMeans(self):
        return self.clusterMeans

#########################Other#Math#Functions#################################
    def calculateDistance(self,point1,point2):
        vecLength = point2 - point1
        return math.sqrt(abs(np.dot(vecLength,vecLength)))
