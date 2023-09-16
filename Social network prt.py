import pandas as pd
import numpy as np

df=pd.read_csv(r'C:\2 NIT\ML\Classification ALGO\KNN\Social_Network_Ads.csv')

x=df.iloc[:,[2,3]].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
ac


from sklearn.metrics import classification_report
cr=classification_report(y_test, y_pred)
cr


bias =knn.score(x_train,y_train)
bias

variance=knn.score(x_test,y_test)
variance

dataset1 =pd.read_csv(r"C:\2 NIT\ML\Classification ALGO\KNN\Social_Network_Ads.csv")
d2 =dataset1.copy()
dataset1 = dataset1.iloc[:,[2,3]].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
m=sc.fit_transform(dataset1)
y_pred1 = pd.DataFrame()


d2 ['y_pred1'] = knn.predict(m)

d2.to_csv('final1.csv')



























