import math
import numpy as np
import pandas as pd
import csv
import tracemalloc
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import time
from sklearn.model_selection import train_test_split
import warnings
import pytest
import sys
import io
warnings.filterwarnings('always')


#Implementation of Decision Tree Classsifier
class Decision_Tree_Classifier:
    
    #Constructor
    def __init__(self, max_depth=2, min_samples_split = 2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
    
    #Calculating GINI score
    def giniScore(self, val):             
        return 1 - sum(((len(val[val == l])/len(val))**2) for l in np.unique(val))
    
    #Calculating Entropy Score
    def entropyScore(self, val):
        entropy = 0
        for l in np.unique(val):
            prob = len(val[val == l]) / len(val)
            entropy = entropy + (-prob * np.log2(prob))        
        return entropy
    
    
    #Calculating infomation Gain
    def informationGain(self, parentNode, leftChild, rightChild, criterion="gini"):     
        
        #If criterion is ENTROPY
        if criterion=="entropy":
            childEntropyAvg = (len(leftChild)/len(parentNode))*self.entropyScore(leftChild) + (len(rightChild)/len(parentNode))*self.entropyScore(rightChild)
            gain = self.entropyScore(parentNode) - childEntropyAvg
        
        #If criterion is GINI
        else:
            childGiniAvg = (len(leftChild)/len(parentNode))*self.giniScore(leftChild) + (len(rightChild)/len(parentNode))*self.giniScore(rightChild)
            gain = self.giniScore(parentNode) - childGiniAvg

        #Return calculated information gain
        return gain
    
    #Returning list with all the midpoints of thresholds given
    def thresholdMidpoints(self, ls):
        mid = []
        for i in range(len(ls)-1):
            mid.append((ls[i]+ls[i+1])/2)
        return mid
            
    
    #Split the data set into left and right leaf
    def bestSplitter(self, X, Y):
        best_feature = None
        best_threshold = None
        best_info_gain = -float('inf')

        for feature in range(X.shape[1]):
            ls = list(set(X[:, feature]))
            thresholds = self.thresholdMidpoints(ls)
            for threshold in thresholds:
                left_indices = [i for i in range(X.shape[0]) if X[i, feature] < threshold] 
                right_indices = [i for i in range(X.shape[0]) if X[i, feature] >= threshold]
                
                if len(left_indices)>0 and len(right_indices)>0:
                    left_Y = Y[left_indices] 
                    right_Y = Y[right_indices]
                    
                
                    if self.criterion == "entropy":
                        information_gain = self.informationGain(Y, left_Y, right_Y, "entropy")
                    elif self.criterion == "gini":
                        information_gain = self.informationGain(Y, left_Y, right_Y, "gini")
                    else:
                        return "Give proper criterion"
                    if information_gain > best_info_gain:
                        best_feature = feature
                        best_threshold = threshold
                        best_info_gain = information_gain

        return best_feature, best_threshold, best_info_gain

    #Generating tree from scratch
    def generateTree(self, X, y, depth=0):
        if depth == self.max_depth or len(set(y)) == 1:
            return y[0]
        
        
        if X.shape[0]>=self.min_samples_split and depth<=self.max_depth:
            feature, threshold, best_info_gain = self.bestSplitter(X, y)
            left_indices = [i for i in range(X.shape[0]) if X[i, feature] < threshold] 
            right_indices = [i for i in range(X.shape[0]) if X[i, feature] >= threshold]
            
            if best_info_gain > 0: 
                left_X = X[left_indices]
                left_y = y[left_indices]

                right_X = X[right_indices]
                right_y = y[right_indices]

                left_subtree = self.generateTree(left_X, left_y, depth + 1)
                right_subtree = self.generateTree(right_X, right_y, depth + 1)

                return (feature, threshold, left_subtree, right_subtree)
        
        ls_y = list(y)
        leaf_value = self.majorityLeafValue(ls_y)
        return (leaf_value)

    
    #Fit which will be called by main a nd instantiate the generateTree
    def fit(self, X, y):
        self.root = self.generateTree(X, y)

    #Select the maximum of class from the list of classes
    def majorityLeafValue(self, Y):
        return max(Y, key=Y.count)
    
    #predicting the test result
    def predict(self, X):
        return [self.makePredictions(inputs) for inputs in X]
    
    #Making predictions based on self.root
    def makePredictions(self, inputs):
        node = self.root
        while isinstance(node, tuple):
            feature, threshold, left, right = node
            if inputs[feature] < threshold:
                node = left
            else:
                node = right
        return node
    
#main Class
def main():
    ###############################################################
#                        IRIS DATASET                         #
###############################################################

# max_depths = [2, 4, 6, 8, 10]
# min_sample_splits = [2, 4, 6, 8, 10]
# criterions = ['gini', 'entropy']


# #==============MANUAL-DECISION TREE CLASSIFIER=================#

# manual_peak_space = []
# manual_time = []
# manual_accuracy = []
# manual_precision = []
# manual_recall = []
# manual_f1 = []
# m_data_size = []
# m_data_feature = []
# m_train_size = []
# m_classes = []
# m_max_depth = []
# m_min_samples_split = []
# m_criterion = []

# df = pd.read_csv("iris.csv", header=None)
# X = df.iloc[:, :-1].values
# Y = df.iloc[:, -1].values
# Y, _dict = pd.Series(Y).factorize()
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.8, random_state=41)



# for cr in criterions:
#     for mss in min_sample_splits:
#         for md in max_depths:
            
#             m_data_size.append(len(X))
            
#             m_data_feature.append(len(df.columns))
            
#             m_train_size.append(len(X_train))
            
#             m_classes.append(len(_dict))
            
#             m_max_depth.append(md)
            
#             m_min_samples_split.append(mss)
            
#             m_criterion.append(cr)
            
#             classifier = Decision_Tree_Classifier(max_depth=md, min_samples_split=mss, criterion=cr)

#             tracemalloc.start()

#             c_time = time.time()
            
#             classifier.fit(X_train,Y_train)

#             c, p = tracemalloc.get_traced_memory()

#             cal_time  = time.time() - c_time

#             manual_time.append(round(cal_time, 5))

#             p = p/(1024*1024)

#             manual_peak_space.append(round(p, 5))

#             tracemalloc.stop()

#             Y_pred = classifier.predict(X_test) 

#             manual_accuracy.append(round(accuracy_score(Y_test, Y_pred),4))
#             manual_precision.append(round(precision_score(Y_test, Y_pred, average='macro'), 4))
#             manual_recall.append(round(recall_score(Y_test, Y_pred, average='macro'), 4))
#             manual_f1.append(round(f1_score(Y_test, Y_pred, average='macro'), 4))


# # #=============SKLEARN-DECISION TREE CLASSIFIER===================#

# sklearn_peak_space = []
# sklearn_time = []
# sklearn_accuracy = []
# sklearn_precision = []
# sklearn_recall = []
# sklearn_f1 = []
# s_data_size = []
# s_data_feature = []
# s_train_size = []
# s_classes = []
# s_max_depth = []
# s_min_samples_split = []
# s_criterion = []

# df = pd.read_csv("iris.csv", header=None)
# X = df.iloc[:, :-1].values
# Y = df.iloc[:, -1].values
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.8, random_state=41)

# for cr in criterions:
#     for mss in min_sample_splits:
#         for md in max_depths:
            
#             s_data_size.append(len(X))
            
#             s_data_feature.append(len(df.columns))
            
#             s_train_size.append(len(X_train))
            
#             s_classes.append(len(_dict))
            
#             s_max_depth.append(md)
            
#             s_min_samples_split.append(mss)
            
#             s_criterion.append(cr)
            
#             classifier = DecisionTreeClassifier(max_depth=md, min_samples_split=mss, criterion=cr)

#             tracemalloc.start()

#             c_time = time.time()
            
#             classifier.fit(X_train,Y_train)

#             c, p = tracemalloc.get_traced_memory()

#             cal_time  = time.time() - c_time

#             sklearn_time.append(round(cal_time, 5))

#             p = p/(1024*1024)

#             sklearn_peak_space.append(round(p, 5))

#             tracemalloc.stop()

#             Y_pred = classifier.predict(X_test) 

#             sklearn_accuracy.append(round(accuracy_score(Y_test, Y_pred),4))
#             sklearn_precision.append(round(precision_score(Y_test, Y_pred, average='macro'), 4))
#             sklearn_recall.append(round(recall_score(Y_test, Y_pred, average='macro'), 4))
#             sklearn_f1.append(round(f1_score(Y_test, Y_pred, average='macro'), 4))


# with open('Evaluation_Report_IRIS.csv', 'w') as file:
#     writer = csv.writer(file)
#     header = ["data_size", "data_feature", "train_size", "max_depth", 
#               "min_samples_split", "criterion", "manual_classes", "sklearn_classes", "manual_time", 
#               "sklearn_time", "manual_peak_space", "sklearn_peak_space", "manual_accuracy", 
#               "sklearn_accuracy", "manual_precision", "sklearn_precision", 
#               "manual_recall", "sklearn_recall", "manual_f1", "sklearn_f1"]
#     writer.writerow(header)
#     for i in range(len(sklearn_peak_space)):
#         writer.writerow([m_data_size[i], m_data_feature[i], m_train_size[i], m_max_depth[i], m_min_samples_split[i], m_criterion[i], m_classes[i], s_classes[i], manual_time[i], sklearn_time[i], manual_peak_space[i], sklearn_peak_space[i], manual_accuracy[i], sklearn_accuracy[i], manual_precision[i], sklearn_precision[i], manual_recall[i], sklearn_recall[i], manual_f1[i], sklearn_f1[i]])
#     file.close()
        

# ###############################################################
# #                ACOUSTIC-FEATURES DATASET                    #
# ###############################################################

# max_depths = [2, 4, 6, 8]
# min_sample_splits = [2, 4]
# criterions = ['gini', 'entropy']


# #=============MANUAL-DECISION TREECLASSIFIER===================#

# manual_peak_space = []
# manual_time = []
# manual_accuracy = []
# manual_precision = []
# manual_recall = []
# manual_f1 = []
# m_data_size = []
# m_data_feature = []
# m_train_size = []
# m_classes = []
# m_max_depth = []
# m_min_samples_split = []
# m_criterion = []

# df = pd.read_csv("Acoustic-Features.csv", skiprows=1, header=None)
# X = df.iloc[:,1:].values
# Y = df.iloc[:,0].values
# Y, _dict = pd.Series(Y).factorize()
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.8, random_state=41)



# for cr in criterions:
#     for mss in min_sample_splits:
#         for md in max_depths:
            
#             m_data_size.append(len(X))
            
#             m_data_feature.append(len(df.columns))
            
#             m_train_size.append(len(X_train))
            
#             m_classes.append(len(_dict))
            
#             m_max_depth.append(md)
            
#             m_min_samples_split.append(mss)
            
#             m_criterion.append(cr)
            
#             classifier = Decision_Tree_Classifier(max_depth=md, min_samples_split=mss, criterion=cr)

#             tracemalloc.start()

#             c_time = time.time()
            
#             classifier.fit(X_train,Y_train)

#             c, p = tracemalloc.get_traced_memory()

#             cal_time  = time.time() - c_time

#             manual_time.append(round(cal_time, 5))

#             p = p/(1024*1024)

#             manual_peak_space.append(round(p, 5))

#             tracemalloc.stop()

#             Y_pred = classifier.predict(X_test) 

#             manual_accuracy.append(round(accuracy_score(Y_test, Y_pred),4))
#             manual_precision.append(round(precision_score(Y_test, Y_pred, average='macro', labels=np.unique(Y_pred)), 4))
#             manual_recall.append(round(recall_score(Y_test, Y_pred, average='macro', labels=np.unique(Y_pred)), 4))
#             manual_f1.append(round(f1_score(Y_test, Y_pred, average='macro', labels=np.unique(Y_pred)), 4))


# #=============SKLEARN-DECISION TREE CLASSIFIER===================#

# sklearn_peak_space = []
# sklearn_time = []
# sklearn_accuracy = []
# sklearn_precision = []
# sklearn_recall = []
# sklearn_f1 = []
# s_data_size = []
# s_data_feature = []
# s_train_size = []
# s_classes = []
# s_max_depth = []
# s_min_samples_split = []
# s_criterion = []

# df = pd.read_csv("Acoustic-Features.csv", skiprows=1, header=None)
# X = df.iloc[:,1:].values
# Y = df.iloc[:,0].values
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.8, random_state=41)

# for cr in criterions:
#     for mss in min_sample_splits:
#         for md in max_depths:
            
#             s_data_size.append(len(X))
            
#             s_data_feature.append(len(df.columns))
            
#             s_train_size.append(len(X_train))
            
#             s_classes.append(len(_dict))
            
#             s_max_depth.append(md)
            
#             s_min_samples_split.append(mss)
            
#             s_criterion.append(cr)
            
#             classifier = DecisionTreeClassifier(max_depth=md, min_samples_split=mss, criterion=cr)

#             tracemalloc.start()

#             c_time = time.time()
            
#             classifier.fit(X_train,Y_train)

#             c, p = tracemalloc.get_traced_memory()

#             cal_time  = time.time() - c_time

#             sklearn_time.append(round(cal_time, 5))

#             p = p/(1024*1024)

#             sklearn_peak_space.append(round(p, 5))

#             tracemalloc.stop()

#             Y_pred = classifier.predict(X_test) 

#             sklearn_accuracy.append(round(accuracy_score(Y_test, Y_pred),4))
#             sklearn_precision.append(round(precision_score(Y_test, Y_pred, average='macro', labels=np.unique(Y_pred)), 4))
#             sklearn_recall.append(round(recall_score(Y_test, Y_pred, average='macro', labels=np.unique(Y_pred)), 4))
#             sklearn_f1.append(round(f1_score(Y_test, Y_pred, average='macro', labels=np.unique(Y_pred)), 4))


# with open('Evaluation_Report_ACOUSTIC.csv', 'w') as file:
#     writer = csv.writer(file)
#     header = ["data_size", "data_feature", "train_size", "max_depth", 
#               "min_samples_split", "criterion", "manual_classes", "sklearn_classes", "manual_time", 
#               "sklearn_time", "manual_peak_space", "sklearn_peak_space", "manual_accuracy", 
#               "sklearn_accuracy", "manual_precision", "sklearn_precision", 
#               "manual_recall", "sklearn_recall", "manual_f1", "sklearn_f1"]
#     writer.writerow(header)
#     for i in range(len(sklearn_peak_space)):
#         writer.writerow([m_data_size[i], m_data_feature[i], m_train_size[i], m_max_depth[i], m_min_samples_split[i], m_criterion[i], m_classes[i], s_classes[i], manual_time[i], sklearn_time[i], manual_peak_space[i], sklearn_peak_space[i], manual_accuracy[i], sklearn_accuracy[i], manual_precision[i], sklearn_precision[i], manual_recall[i], sklearn_recall[i], manual_f1[i], sklearn_f1[i]])
#     file.close()
        





# ###############################################################
# #                  SEGMENTATION DATASET                       #
# ###############################################################

# max_depths = [2, 4, 6, 8]
# min_sample_splits = [2, 4]
# criterions = ['gini', 'entropy']

# #=============MANUAL-DECISION TREECLASSIFIER===================#

# manual_peak_space = []
# manual_time = []
# manual_accuracy = []
# manual_precision = []
# manual_recall = []
# manual_f1 = []
# m_data_size = []
# m_data_feature = []
# m_train_size = []
# m_classes = []
# m_max_depth = []
# m_min_samples_split = []
# m_criterion = []

# df = pd.read_csv("segmentation.csv", skiprows=1, header=None)
# X = df.iloc[:,1:].values
# Y = df.iloc[:,0].values
# Y, _dict = pd.Series(Y).factorize()
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.8, random_state=41)



# for cr in criterions:
#     for mss in min_sample_splits:
#         for md in max_depths:
            
#             m_data_size.append(len(X))
            
#             m_data_feature.append(len(df.columns))
            
#             m_train_size.append(len(X_train))
            
#             m_classes.append(len(_dict))
            
#             m_max_depth.append(md)
            
#             m_min_samples_split.append(mss)
            
#             m_criterion.append(cr)
            
#             classifier = Decision_Tree_Classifier(max_depth=md, min_samples_split=mss, criterion=cr)

#             tracemalloc.start()

#             c_time = time.time()
            
#             classifier.fit(X_train,Y_train)

#             c, p = tracemalloc.get_traced_memory()

#             cal_time  = time.time() - c_time

#             manual_time.append(round(cal_time, 5))

#             p = p/(1024*1024)

#             manual_peak_space.append(round(p, 5))

#             tracemalloc.stop()

#             Y_pred = classifier.predict(X_test) 

#             manual_accuracy.append(round(accuracy_score(Y_test, Y_pred),4))
#             manual_precision.append(round(precision_score(Y_test, Y_pred, average='macro', labels=np.unique(Y_pred)), 4))
#             manual_recall.append(round(recall_score(Y_test, Y_pred, average='macro', labels=np.unique(Y_pred)), 4))
#             manual_f1.append(round(f1_score(Y_test, Y_pred, average='macro', labels=np.unique(Y_pred)), 4))


# #=============SKLEARN-DECISION TREE CLASSIFIER===================#

# sklearn_peak_space = []
# sklearn_time = []
# sklearn_accuracy = []
# sklearn_precision = []
# sklearn_recall = []
# sklearn_f1 = []
# s_data_size = []
# s_data_feature = []
# s_train_size = []
# s_classes = []
# s_max_depth = []
# s_min_samples_split = []
# s_criterion = []

# df = pd.read_csv("segmentation.csv", skiprows=1, header=None)
# X = df.iloc[:,1:].values
# Y = df.iloc[:,0].values
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.8, random_state=41)

# for cr in criterions:
#     for mss in min_sample_splits:
#         for md in max_depths:
            
#             s_data_size.append(len(X))
            
#             s_data_feature.append(len(df.columns))
            
#             s_train_size.append(len(X_train))
            
#             s_classes.append(len(_dict))
            
#             s_max_depth.append(md)
            
#             s_min_samples_split.append(mss)
            
#             s_criterion.append(cr)
            
#             classifier = DecisionTreeClassifier(max_depth=md, min_samples_split=mss, criterion=cr)

#             tracemalloc.start()

#             c_time = time.time()
            
#             classifier.fit(X_train,Y_train)

#             c, p = tracemalloc.get_traced_memory()

#             cal_time  = time.time() - c_time

#             sklearn_time.append(round(cal_time, 5))

#             p = p/(1024*1024)

#             sklearn_peak_space.append(round(p, 5))

#             tracemalloc.stop()

#             Y_pred = classifier.predict(X_test) 

#             sklearn_accuracy.append(round(accuracy_score(Y_test, Y_pred),4))
#             sklearn_precision.append(round(precision_score(Y_test, Y_pred, average='macro', labels=np.unique(Y_pred)), 4))
#             sklearn_recall.append(round(recall_score(Y_test, Y_pred, average='macro', labels=np.unique(Y_pred)), 4))
#             sklearn_f1.append(round(f1_score(Y_test, Y_pred, average='macro', labels=np.unique(Y_pred)), 4))


# with open('Evaluation_Report_SEGMENTATION.csv', 'w') as file:
#     writer = csv.writer(file)
#     header = ["data_size", "data_feature", "train_size", "max_depth", 
#               "min_samples_split", "criterion", "manual_classes", "sklearn_classes", "manual_time", 
#               "sklearn_time", "manual_peak_space", "sklearn_peak_space", "manual_accuracy", 
#               "sklearn_accuracy", "manual_precision", "sklearn_precision", 
#               "manual_recall", "sklearn_recall", "manual_f1", "sklearn_f1"]
#     writer.writerow(header)
#     for i in range(len(sklearn_peak_space)):
#         writer.writerow([m_data_size[i], m_data_feature[i], m_train_size[i], m_max_depth[i], m_min_samples_split[i], m_criterion[i], m_classes[i], s_classes[i], manual_time[i], sklearn_time[i], manual_peak_space[i], sklearn_peak_space[i], manual_accuracy[i], sklearn_accuracy[i], manual_precision[i], sklearn_precision[i], manual_recall[i], sklearn_recall[i], manual_f1[i], sklearn_f1[i]])
#     file.close()
    
    
    
#==================PYTEST-TXT FILES=======================#




    if __name__ == '__main__':

        so = sys.stdout 

        sys.stdout = io.StringIO()

        pytest.main()

        output = sys.stdout.getvalue()

        sys.stdout.close()

        sys.stdout = so 



        try:     

            file = open("test-output.txt", "w")     

            file.write(output)     

            file.close()     

            print('Saving the logs') 

        except:     

            print('Error in saving logs')



if __name__ == '__main__':
    main()
