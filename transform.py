import pandas as pd 
import pickle

with open('./models/df3.pkl','rb') as f:
       df = pickle.load(f)

df = df.drop('survived',axis=1)

data = {
    'Pclass':df.pclass.unique(),
    'sex' : df.sex.unique(),
    'Embarked':df.embarked.unique(),
    'Title':df.title.unique(),
    'Family_size':df.family_size.unique(),
    'fare':df.fare.unique()
}





print("working")
# print(data)
# print(df.fare.min())
# print(df.fare.max())

if __name__ =="__main__":
    pass

# df = pd.DataFrame(df_pack)

