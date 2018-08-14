import pandas as pd
from sklearn.preprocessing import scale
from sklearn import preprocessing,cross_validation
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

data=pd.read_csv("credit_data.csv",index_col='ID')
data.columns=data.columns.str.lower()
df=data[['limit_bal','bill_amt1','bill_amt2','bill_amt3','bill_amt4','bill_amt5','bill_amt6','pay_amt1','pay_amt1','pay_amt2','pay_amt3','pay_amt4','pay_amt5','pay_amt6']]
df=scale(df)
X=np.array(data.drop(['default.payment.next.month'],1))
X=preprocessing.scale(X)
Y=np.array(data['default.payment.next.month'])
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.2)

clf=LogisticRegression()
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)




