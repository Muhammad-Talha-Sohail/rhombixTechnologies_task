import pickle
import pandas as pd

# Preprocessing
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,ShuffleSplit

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Making pipelines
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer


# Making Matrices
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



#  Apply Column Transformer
with open('./models/column_transformer.pkl','rb') as f:
       ct = pickle.load(f)


with open('./models/LR_model.pkl','rb') as f:
       LR = pickle.load(f)

with open('./models/RFC_model.pkl','rb') as f:
       RFC = pickle.load(f)

with open('./models/SVC_model.pkl','rb') as f:
       SVC = pickle.load(f)




class Data_Model():
    def __init__(self,user_input):
       self.user_input = user_input
       
    def columnTransform(self):
        test_data = pd.DataFrame(self.user_input) 
        # print("transform  is :",test_data)
        return  ct.transform(test_data) 
        
    def LRmodel(self,test):
        
        return LR.predict(test) 
    def RFCmodel(self,test):
   
      return RFC.predict(test) 
  
    def SVCmodel(self,test):
   
      return SVC.predict(test) 
        
    
    


 
       
       
    