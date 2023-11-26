import pandas
from sklearn.model_selection import train_test_split
from sklearn import tree, neighbors
from sklearn.ensemble import RandomForestClassifier 
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, fowlkes_mallows_score, jaccard_score, rand_score
import random
random.seed(9)


#Decision Tree Classifier
def DecisionTree(x_train, y_train, x_test):
    classifier = tree.DecisionTreeClassifier()
    return Classification(classifier, x_train, y_train, x_test)

#Stochastic Gradient Descent
def SGD(x_train, y_train, x_test):
    classifier = linear_model.SGDClassifier(max_iter=5, tol=None)
    return Classification(classifier, x_train, y_train, x_test)

#Support Vector Machine
def SVM(x_train, y_train, x_test):
    classifier = LinearSVC()
    return Classification(classifier, x_train, y_train, x_test)

#KNN Classifier
def KNN(x_train, y_train, x_test):
    classifier = neighbors.KNeighborsClassifier()
    return Classification(classifier, x_train, y_train, x_test)

#Gaussian Naive Bayes
def NaiveBayes(x_train, y_train, x_test):
    classifier = GaussianNB()
    return Classification(classifier, x_train, y_train, x_test)

#Random Forest Classifier
def RandomForest(x_train, y_train, x_test):
    classifier = RandomForestClassifier()
    return Classification(classifier, x_train, y_train, x_test)

#Classification
def Classification(classifier, x_train, y_train, x_test):
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    return predictions

#Scores
def Scores(y_test, predictions):
    Acc = accuracy_score(y_test, predictions)
    F1Score = f1_score(y_test, predictions, average='micro')
    JScore = jaccard_score(y_test, predictions, average='micro')
    FMScore = fowlkes_mallows_score(y_test, predictions)
    RI = rand_score(y_test, predictions)
    return Acc, F1Score, JScore, FMScore, RI

Dataset = pandas.read_csv('Iris.csv')
content = pandas.DataFrame()
attr = len(Dataset.columns)
datapoints = len(Dataset)
x = pandas.DataFrame()
for i in range(1, attr-1, 1):
    ColName = str(Dataset.columns[i])
    Col = Dataset.iloc[:, i]
    x[ColName] = Col
y = Dataset.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

print("\nModel: Decision Tree Classifier")
DTPred = DecisionTree(x_train, y_train, x_test)
content['Decision Tree Predictions'] = DTPred
DTAcc, DTF1Score, DTJScore, DTFMScore, DTRI = Scores(y_test, DTPred)
print("Accuracy: ", DTAcc)
print("F1 Score: ", DTF1Score)
print("Jaccard Score: ", DTJScore)
print("Fowlkes Mallows Score: ", DTFMScore)
print("Rand Index: ", DTRI)

print("\nModel: K Nearest Neighbours Classifier")
KNNPred = KNN(x_train, y_train, x_test)
content['K Nearest Neighbours Predictions'] = KNNPred
KNNAcc, KNNF1Score, KNNJScore, KNNFMScore, KNNRI = Scores(y_test, KNNPred)
print("Accuracy: ", KNNAcc)
print("F1 Score: ", KNNF1Score)
print("Jaccard Score: ", KNNJScore)
print("Fowlkes Mallows Score: ", KNNFMScore)
print("Rand Index: ", KNNRI)

print("\nModel: Naive Bayes Classifier")
NBPred = NaiveBayes(x_train, y_train, x_test)
content['Naive Bayes Predictions'] = NBPred
NBAcc, NBF1Score, NBJScore, NBFMScore, NBRI = Scores(y_test, NBPred)
print("Accuracy: ", NBAcc)
print("F1 Score: ", NBF1Score)
print("Jaccard Score: ", NBJScore)
print("Fowlkes Mallows Score: ", NBFMScore)
print("Rand Index: ", NBRI)

print("\nModel: Random Forest Classifier")
RFPred = RandomForest(x_train, y_train, x_test)
content['Random Forest Predictions'] = RFPred
RFAcc, RFF1Score, RFJScore, RFFMScore, RFRI = Scores(y_test, RFPred)
print("Accuracy: ", RFAcc)
print("F1 Score: ", RFF1Score)
print("Jaccard Score: ", RFJScore)
print("Fowlkes Mallows Score: ", RFFMScore)
print("Rand Index: ", RFRI)

print("\nModel: Stochastic Gradient Descent")
SGDPred = SGD(x_train, y_train, x_test)
content['Stochastic Gradient Descent Predictions'] = SGDPred
SGDAcc, SGDF1Score, SGDJScore, SGDFMScore, SGDRI = Scores(y_test, SGDPred)
print("Accuracy: ", SGDAcc)
print("F1 Score: ", SGDF1Score)
print("Jaccard Score: ", SGDJScore)
print("Fowlkes Mallows Score: ", SGDFMScore)
print("Rand Index: ", SGDRI)

print("\nModel: Support Vector Machine Classifier")
SVMPred = SVM(x_train, y_train, x_test)
content['Support Vector Machine Predictions'] = SVMPred
SVMAcc, SVMF1Score, SVMJScore, SVMFMScore, SVMRI = Scores(y_test, SVMPred)
print("Accuracy: ", SVMAcc)
print("F1 Score: ", SVMF1Score)
print("Jaccard Score: ", SVMJScore)
print("Fowlkes Mallows Score: ", SVMFMScore)
print("Rand Index: ", SVMRI)

Details = [""]*len(DTPred)
DT = [""]*len(DTPred)
KNN = [""]*len(DTPred)
NB = [""]*len(DTPred)
RF = [""]*len(DTPred)
SGD = [""]*len(DTPred)
SVM = [""]*len(DTPred)

Details[0] = "Metric"
Details[1] = "Accuracy"
Details[2] = "F1 Score"
Details[3] = "Jaccard Score"
Details[4] = "Fowlkes Mallows Score"
Details[5] = "Rand Index"

DT[1] = DTAcc
DT[2] = DTF1Score
DT[3] = DTJScore
DT[4] = DTFMScore
DT[5] = DTRI

KNN[1] = KNNAcc
KNN[2] = KNNF1Score
KNN[3] = KNNJScore
KNN[4] = KNNFMScore
KNN[5] = KNNRI

NB[1] = NBAcc
NB[2] = NBF1Score
NB[3] = NBJScore
NB[4] = NBFMScore
NB[5] = NBRI

RF[1] = RFAcc
RF[2] = RFF1Score
RF[3] = RFJScore
RF[4] = RFFMScore
RF[5] = RFRI

SGD[1] = SGDAcc
SGD[2] = SGDF1Score
SGD[3] = SGDJScore
SGD[4] = SGDFMScore
SGD[5] = SGDRI

SVM[1] = SVMAcc
SVM[2] = SVMF1Score
SVM[3] = SVMJScore
SVM[4] = SVMFMScore
SVM[5] = SVMRI

content["Classifier"] = Details
content["Decision Tree Classifier"] = DT
content["K Nearest Neighbours Classifier"] = KNN
content["Naive Bayes Classifier"] = NB
content["Random Forest Classifier"] = RF
content["Stochastic Gradient Descent"] = SGD
content["Support Vector Machine"] = SVM

content.to_csv('Iris Classification.csv', index=False)
