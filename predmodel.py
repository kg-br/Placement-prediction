import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
data=pd.read_csv("Placement_Data_Full_Class.csv")

data=data.rename(columns={'ssc_p':"ssc percentage",'ssc_b':"ssc board",'hsc_p':"hsc percentage",'hsc_b':"hsc board"
                         ,'hsc_s':"hsc stream",'degree_p':"degree percentage",'degree_t':'degree stream',
                          'workex':"work experience",'etest_p':"apptitude_score",'mba_p':'mba_percentage'})
data=data.iloc[:,:-1]

data.drop(['sl_no'], axis=1, inplace = True)

le = preprocessing.LabelEncoder()
data['work experience']=le.fit_transform(data['work experience'])
data['status']=le.fit_transform(data['status'])

x=data.iloc[:,[1,6,8,11]].values
y=data.iloc[:,12].values
x_ttrain,x_ttest,y_ttrain,y_ttest=train_test_split(x,y,test_size=0.3,random_state=20)
modelfinal = RandomForestClassifier(n_estimators=100)
modelfinal.fit(x_ttrain, y_ttrain)


inputt=[float(x) for x in "73.5 88.61 76.8 1".split(' ')]
final=[np.array(inputt)]

b = modelfinal.predict(final)

pickle.dump(modelfinal,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))