import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import gridspec 


from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler

#Github

#Decision Tree Classifier
def DecisionTree(X_train, Y_train, X_test, Y_test):
    classifier = DecisionTreeClassifier() 
    return Classification(classifier, X_train, Y_train, X_test, Y_test)

#Stochastic Gradient Descent
def SGD(X_train, Y_train, X_test, Y_test):
    classifier = linear_model.SGDClassifier(max_iter=5, tol=None)
    return Classification(classifier, X_train, Y_train, X_test, Y_test)

#Support Vector Machine
def SVM(X_train, Y_train, X_test, Y_test):
    classifier = LinearSVC()
    return Classification(classifier, X_train, Y_train, X_test, Y_test)

#KNN Classifier
def KNN(X_train, Y_train, X_test, Y_test):
    classifier = KNeighborsClassifier()
    return Classification(classifier, X_train, Y_train, X_test, Y_test)

#Gaussian Naive Bayes
def NaiveBayes(X_train, Y_train, X_test, Y_test):
    classifier = GaussianNB()
    return Classification(classifier, X_train, Y_train, X_test, Y_test)

#Random Forest Classifier
def RandomForest(X_train, Y_train, X_test, Y_test):
    classifier = RandomForestClassifier()
    return Classification(classifier, X_train, Y_train, X_test, Y_test)

#Classification
def Classification(classifier, X_train, Y_train, X_test, Y_test):
    classifier.fit(X_train, Y_train)
    predictions = classifier.predict(X_test)
    report = classification_report(Y_test, predictions)
    return predictions, report

def Preprocess(X):
    # Irrelevant columns
    del_col=['merchant','first','last','street','zip','unix_time','Unnamed: 0','trans_num','cc_num']
    X.drop(columns=del_col,inplace=True)
   
    # Data Conversion 
    X['trans_date_trans_time']=pd.to_datetime(X['trans_date_trans_time'])
    X['trans_date']=X['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
    X['trans_date']=pd.to_datetime(X['trans_date'])
    X['dob']=pd.to_datetime(X['dob'])
    
    #Calculate Age of each transaction
    X["age"] = (X["trans_date"] - X["dob"]).dt.days //365
    X['trans_month']=X['trans_date'].dt.month
    X['trans_year']=X['trans_date'].dt.year
    
    # The gender column is transformed to binary values
    X['gender']=X['gender'].apply(lambda x : 1 if x=='M' else 0)
    X['gender']=X['gender'].astype(int)
    
    # Distance Calculated
    X['lat_dis']=abs(X['lat']-X['merch_lat'])
    X['long_dis']=abs(X['long']-X['merch_long'])
    
    X=pd.get_dummies(X,columns=['category'])
    X=X.drop(columns=['city','trans_date_trans_time','state','job','merch_lat','merch_long','lat','long','dob','trans_date'])
    return X


test_df = pd.read_csv("fraudTest.csv")
train_df = pd.read_csv("fraudTrain.csv")
print(train_df.info())

cnt =train_df['is_fraud'].value_counts()
prp = train_df['is_fraud'].value_counts(normalize =True)*100
t = pd.concat([cnt,prp],axis=1)
t.index=['Genuine','Fraud']
prp.plot(kind='bar')
plt.show()

#Seeing trends of frauds

cat_counts = train_df.groupby(['category','is_fraud'])['is_fraud'].count().unstack()
print(cat_counts)
cat_counts_fraud = cat_counts[1]
ccc = cat_counts_fraud.plot(kind='bar', figsize=(12, 6))
ccc.set_ylabel('No. od Frauds')
ccc.set_xlabel('Category')
ccc.set_title('Category Vs Frauds')
plt.xticks(rotation=45)
plt.show()

gen_counts = train_df[train_df['is_fraud'] == 1].groupby('gender')['is_fraud'].count()
print(gen_counts)
ax = gen_counts.plot(kind='bar')
ax.set_xlabel('Gender')
ax.set_ylabel('Fraud Count')
ax.set_title('Gender Vs Fraud')
plt.show()

df_zip = train_df[train_df['is_fraud'] == 1].groupby('zip')['is_fraud'].count()
top_10_zip= df_zip.sort_values(ascending=False).head(10)
print(top_10_zip)

df_city = train_df[train_df['is_fraud'] == 1].groupby('city')['is_fraud'].count()
top_10_city= df_city.sort_values(ascending=False).head(10)
print(top_10_city)

df_mer = train_df[train_df['is_fraud'] == 1].groupby('merchant')['is_fraud'].count()
top_10_mer= df_mer.sort_values(ascending=False).head(10)
print(top_10_mer)

train_df_pre = Preprocess(train_df.copy())
test_df_pre = Preprocess(test_df.copy())
print(train_df_pre.info())

X_train=train_df_pre.drop('is_fraud',axis=1)
Y_train=train_df_pre['is_fraud']
X_test=test_df_pre.drop('is_fraud',axis=1)
Y_test=test_df_pre['is_fraud']

scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
content = pd.DataFrame()
file = open("Credit Card Fraud Detection Prediction Reports.txt", 'w')

file.write("Decision Tree \n")
DTPred, DTRep = DecisionTree(X_train, Y_train, X_test, Y_test)
content['Decision Tree Predictions'] = DTPred
file.write(DTRep)
file.write("\n")
print("Decision Tree: \n")
print(DTRep)

file.write("K Nearest Neighbours \n")
KNNPred, KNNRep = KNN(X_train, Y_train, X_test, Y_test)
content['K Nearest Neighbours Predictions'] = KNNPred
file.write(KNNRep)
file.write("\n")
print("K Nearest Neighbours: \n")
print(KNNRep)

file.write("Naive Bayes \n")
NBPred, NBRep = NaiveBayes(X_train, Y_train, X_test, Y_test)
content['Naive Bayes Predictions'] = NBPred
file.write(NBRep)
file.write("\n")
print("Naive Bayes: \n")
print(NBRep)

file.write("Random Forest \n")
RFPred, RFRep = RandomForest(X_train, Y_train, X_test, Y_test)
content['Random Forest Predictions'] = RFPred
file.write(RFRep)
file.write("\n")
print("Random Forest: \n")
print(RFRep)

file.write("Stochastic Gradient Descent \n")
SGDPred, SGDRep = SGD(X_train, Y_train, X_test, Y_test)
content['Stochastic Gradient Descent Predictions'] = SGDPred
file.write(SGDRep)
file.write("\n")
print("Stochastic Gradient Descent: \n")
print(SGDRep)

file.write("Support Vector Machine \n")
SVMPred, SVMRep = SVM(X_train, Y_train, X_test, Y_test)
content['Support Vector Machine Predictions'] = SVMPred
file.write(SVMRep)
file.write("\n")
print("Support Vector Machine: \n")
print(SVMRep)

file.close()
content.to_csv('Credit Card Fraud Detection Prediction.csv', index=False)
