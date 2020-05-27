# app.py
from flask import Flask           # import flask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import jsonify
from flask import request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json
from flask_cors import CORS     
app = Flask(__name__)             # create an app instance
CORS(app)

# dataset = pd.read_csv('data1.csv')

# dataset['sex']=dataset['sex'].replace(['female','male','other'],[0,1,2])
# dataset['smoking']=dataset['smoking'].replace(['never','quit','yes'],[0,1,2])
# dataset['working']=dataset['working'].replace(['home','never','stopped','travel critical','travel non critical'],[1,0,2,3,4])
# dataset['income']=dataset['income'].replace(['blank','gov','high','med','low'],[0,4,3,2,1])
# dataset['blood_type']=dataset['blood_type'].replace(['unknown','abn','abp','an','ap','bn','bp','on','op'],[0,1,2,3,4,5,6,7,8])
# dataset['insurance']=dataset['insurance'].replace(['no','yes'],[0,1])

# X1=dataset.iloc[:,[0,1,2,3,4,5,6,9,10,12,18,19,20,21,26,32]].values
# y1=dataset.iloc[:,33].values
# X2=dataset.iloc[:,0:33].values
# y2=dataset.iloc[:,34].values

# from sklearn.model_selection import train_test_split
# X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.25,random_state=42)
# X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.25,random_state=42)

# from sklearn.linear_model import LinearRegression
# regressor1=LinearRegression()
# regressor2=LinearRegression()
# regressor1.fit(X_train1,y_train1)
# regressor2.fit(X_train2,y_train2)

# y_pred1 = regressor1.predict(X_test1)
# y_pred2 = regressor2.predict(X_test2)
# print("Accuracy for covid risk training set",regressor1.score(X_train1,y_train1)*100,"%")
# print("Accuracy for covid risk test set",regressor1.score(X_test1,y_test1)*100,"%")
# print("Accuracy for death risk training set",regressor2.score(X_train2,y_train2)*100,"%")
# print("Accuracy for death risk test set",regressor2.score(X_test2,y_test2)*100,"%")

# //Old  one
# dataset = pd.read_csv('data2.csv')


# dataset['sex']=dataset['sex'].replace(['female','male','other'],[0,1,2])
# dataset['smoking']=dataset['smoking'].replace(['never','quit','yes'],[0,1,2])
# dataset['working']=dataset['working'].replace(['home','never','stopped','travel critical','travel non critical'],[1,0,2,3,4])
# dataset['income']=dataset['income'].replace(['blank','gov','high','med','low'],[0,4,3,2,1])

# X=dataset.iloc[:,0:16].values
# y=dataset.iloc[:,16].values

# '''
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0,4,5,9])],remainder='passthrough')
# X= np.array(ct.fit_transform(X), dtype=np.float)

# X=X[:,[0,1,3,4,5,6,8,9,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27]]'''


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=42)


# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# from sklearn.linear_model import LinearRegression
# regressor=LinearRegression()
# regressor.fit(X_train,y_train)

# y_pred = regressor.predict(X_test)
# print("Accuracy for training set",regressor.score(X_train,y_train)*100)
# print("Accuracy for test set",regressor.score(X_test,y_test)*100)


# @app.route("/search")                   # at the end point /
# def hello():
        
#         # print(int(request.args.get('gen')))
#         gender=int(request.args.get('gen'))
#         age=int(request.args.get('age'))
#         height=int(request.args.get('hgt'))
#         weight=int(request.args.get('wgt'))
#         income=int(request.args.get('inc'))
#         smoking=int(request.args.get('smk'))
#         alcohol=int(request.args.get('alc'))
#         contacts=int(request.args.get('con'))
#         totalpeople=int(request.args.get('totpep'))
#         working=int(request.args.get('wrkng'))
#         masks=int(request.args.get('masks'))
#         symptoms=int(request.args.get('sym'))
#         contactsinfected=int(request.args.get('coninf'))
#         asthma=int(request.args.get('asthma'))
#         lung=int(request.args.get('lng'))
#         healthworker=int(request.args.get('hlth'))

#         new_input=np.array([gender,age,height,weight,income,smoking,alcohol,contacts,totalpeople,working,masks,symptoms,contactsinfected,asthma,lung,healthworker])
#         new_input1=new_input.reshape(1,-1)
#         new_output = regressor.predict(new_input1)
#         print("\nRisk of dying from covid-19", new_output/100,"%")

#         freqs = {
#         'predictedoutput': new_output[0]/100
#         # 'accuracy': result,
#         }
#         return  jsonify(freqs)  
#              # which returns "hello world"
#               # username = request.args.get('username')
#         # password = request.args.get('password') 

#NEW FINAL

dataset = pd.read_csv('datasetfinal.csv')


dataset['sex']=dataset['sex'].replace(['female','male','other'],[0,1,2])
dataset['smoking']=dataset['smoking'].replace(['never','quit','yes'],[0,1,2])
dataset['working']=dataset['working'].replace(['home','never','stopped','travel critical','travel non critical'],[1,0,2,3,4])
dataset['income']=dataset['income'].replace(['blank','gov','high','med','low'],[0,4,3,2,1])
dataset['blood_type']=dataset['blood_type'].replace(['unknown','abn','abp','an','ap','bn','bp','on','op'],[0,1,2,3,4,5,6,7,8])
dataset['insurance']=dataset['insurance'].replace(['blank','no','yes'],[0,1,2])
dataset['race']=dataset['race'].replace(['asian','black','hispanic','mixed','other','white'],[1,2,3,4,5,6])
dataset['immigrant']=dataset['immigrant'].replace(['immigrant','native'],[0,1])

X=dataset.iloc[:,0:39].values
y1=dataset.iloc[:,39].values
y2=dataset.iloc[:,40].values


from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size = 0.25,random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size = 0.25,random_state=42)

from sklearn.linear_model import LinearRegression
regressor1=LinearRegression()
regressor2=LinearRegression()
regressor1.fit(X_train1,y_train1)
regressor2.fit(X_train2,y_train2)

y_pred1 = abs(regressor1.predict(X_test1))
y_pred2 = abs(regressor2.predict(X_test2))
print("Accuracy for covid risk training set",regressor1.score(X_train1,y_train1)*100,"%")
print("Accuracy for covid risk test set",regressor1.score(X_test1,y_test1)*100,"%")
print("Accuracy for death risk training set",regressor2.score(X_train2,y_train2)*100,"%")
print("Accuracy for death risk test set",regressor2.score(X_test2,y_test2)*100,"%")
# #put url here in the arrays
# #covid risk
# new_input_covid=np.array([[[gender,age,height,weight,income,smoking,alcohol,blood_type,insurance,contacts_count,housecount,public_transport,working,worried,self_reducing_risk,self_social_distancing,self_washing_hands,house_reducing_risk,house_social_distancing,house_washing_hands,sanitizer,masks,symptoms,contacts_infected,asthma,kidney_disease,liver_disease,compromised_immune,heart_disease,lung,diabetes,hiv_positive,hypertension,other_chronic,nursing_home,health_worker]]])
# new_input_covid1=new_input_covid.reshape(1,-1)
# new_output_covid= regressor1.predict(new_input_covid1)
# print("\nRisk of getting covid-19", abs(new_output_covid),"%")

# #death risk
# new_input_death=np.array([[[gender,age,height,weight,income,smoking,alcohol,blood_type,insurance,contacts_count,housecount,public_transport,working,worried,self_reducing_risk,self_social_distancing,self_washing_hands,house_reducing_risk,house_social_distancing,house_washing_hands,sanitizer,masks,symptoms,contacts_infected,asthma,kidney_disease,liver_disease,compromised_immune,heart_disease,lung,diabetes,hiv_positive,hypertension,other_chronic,nursing_home,health_worker]]])
# new_input_death1=new_input_death.reshape(1,-1)
# new_output_death= regressor2.predict(new_input_death1)
# print("\nRisk of dying from covid-19", abs(new_output_death),"%")

@app.route("/deathandrisk")                   # at the end point /
def deathandrisk():
        # sex=Enter 0/1/2 for female/male/other
        # age=Enter age
        # height=Enter height in cms
        # weight=Enter weight in kgs
        # bmi=enter bmi
        # blood_type= Enter 0/1/2/3/5/6/7/8 for your blood type- unknown/abn/abp/an/ap/bn/bp/on/op
        # insurance=Enter 0/1 if you- dont have/have insurance
        # income=Enter 0/1/2/3/4 if- no income/low/med/high/gov income
        # race= Enter 1/2/3/4/5/6 if youre asian/black/hispanic/mixed/other/white
        # immigrant= Enter 0/1 if youre immigrant/native
        # smoking=Enter 0/1/2 for smoking- never/quit/yes
        # alcohol=Enter number of times you consumed alcohol in last 2 weeks, -1 if never consumed 
        # contacts_count=Enter total contacts(around 5)
        # house_count=Enter total people in your house
        # public_transport=Enter total number of times you used public transport in the last week
        # working=Enter 0/1/2/3/4 for working options- never/home/stopped/travel critical/travel non critical
        # worried= Enter -2/-1/0/1/2 if you're not worried at all/not worried/neutral/very worried/very very worried
        # self_reducing_risk=Enter -2/-1/0/1/2 if you never/not at all/neutral/yes/definitely taking steps to reduce risk
        # self_social_distancing= Enter -2/-10/1/2 if you- never/avoid/neutral/sometimes/regularly follow social distancing 
        # self_washing_hands= Enter -2/-1/0/1/2 if you- never/dont/neutral/sometimes/regularly wash your hands
        # house_reducing_risk=Enter -2/-1/0/1/2 if your family never/not at all/neutral/yes/definitely are taking steps to reduce risk
        # house_social_distancing= Enter -2/-1/0/1/2 if your family - never/doesnt/neutral/sometimes/regulary follow social distancing 
        # house_washing_hands=Enter -2/-1/0/1/2 if your family- never/do not/neutral/sometimes/regularly wash hands
        # sanitizer= Enter -2/-1/0/1/2 if you- never/dont/neutral/sometimes/regularly use sanitizer
        # masks=On a scale of 0-5, tell us how serious you are about using masks outside
        # symptoms=Enter 0 if you have no covid19 symptoms else 1
        # contacts_infected= Enter 0 if you have not been in contact with an infected person else 1
        # asthma=Enter 0 if you dont have asthma else 1
        # kidney_disease= Enter 0 if you dont have any kindey disease else 1
        # liver_disease=Enter 0 if you dont have any liver disease else 1
        # compromised_immune=Enter 0 if you dont have a compromised immune system else 1
        # heart_disease=Enter 0 if you dont have any heart disease else 1
        # lung_disease=Enter 0 if you have dont have any other lung disease else 1
        # diabetes=Enter 0 if you have dont have diabetes else 1
        # hiv_positive=Enter 0 if you are not HIV positive else 1
        # hypertension=Enter 0 if you have dont have hypertension else 1
        # other_chronic=Enter 0 if you have dont have any other chronic disease else 1
        # nursing_home=Enter 0 if you dont work in a nursing home else 1
        # health_worker=Enter 0 if you're not a health worker else 1
        #################

        # bmi= int(request.args.get('bmi'))
        # race=int(request.args.get('race'))
        # immigrant=int(request.args.get('img'))
        # worried=int(request.args.get('wrd'))
        # self_reducing_risk=int(request.args.get('sredris'))
        # house_reducing_risk=int(request.args.get('hredris'))

        sex = int(request.args.get('gen'))
        age = int(request.args.get('age'))
        height = int(request.args.get('hgt'))
        weight = int(request.args.get('wgt'))
        bmi= int(request.args.get('bmi'))
        income = int(request.args.get('inc'))
        race=int(request.args.get('race'))
        immigrant=int(request.args.get('img'))
        smoking = int(request.args.get('smk'))
        alcohol = int(request.args.get('alc'))
        blood_type = int(request.args.get('bt'))
        insurance = int(request.args.get('ins'))
        contacts_count = int(request.args.get('concnt'))
        housecount = int(request.args.get('hcnt'))
        public_transport = int(request.args.get('ptrnspt'))
        working = int(request.args.get('wrkng'))
        worried=int(request.args.get('wrd'))
        self_reducing_risk=int(request.args.get('sredris'))
        self_social_distancing = int(request.args.get('ssocdis'))
        self_washing_hands = int(request.args.get('swashnds'))
        house_reducing_risk=int(request.args.get('hredris'))
        house_social_distancing = int(request.args.get('hsocdis'))
        house_washing_hands = int(request.args.get('hwashnds'))
        sanitizer = int(request.args.get('san'))
        masks = int(request.args.get('masks'))
        symptoms = int(request.args.get('sym'))
        contacts_infected = int(request.args.get('coninf'))
        asthma = int(request.args.get('asthma'))
        kidney_disease = int(request.args.get('kidney'))
        liver_disease = int(request.args.get('liver'))
        compromised_immune = int(request.args.get('cmpimm'))
        heart_disease = int(request.args.get('heart'))
        lung_disease = int(request.args.get('lung'))
        diabetes = int(request.args.get('diabetes'))
        hiv_positive = int(request.args.get('hiv'))
        hypertension = int(request.args.get('hyprt'))
        other_chronic = int(request.args.get('otherchr'))
        nursing_home = int(request.args.get('nrsng'))
        health_worker = int(request.args.get('hwork'))

        #covid risk
        new_input_covid=np.array([[[sex,age,height,weight,bmi,blood_type,insurance,income,race,immigrant,smoking,alcohol,contacts_count,housecount,public_transport,working,worried,self_reducing_risk,self_social_distancing,self_washing_hands,house_reducing_risk,house_social_distancing,house_washing_hands,sanitizer,masks,symptoms,contacts_infected,asthma,kidney_disease,liver_disease,compromised_immune,heart_disease,lung_disease,diabetes,hiv_positive,hypertension,other_chronic,nursing_home,health_worker]]])
        new_input_covid1=new_input_covid.reshape(1,-1)
        new_output_covid= regressor1.predict(new_input_covid1)
        print("\nRisk of getting covid-19", abs(new_output_covid),"%")

        #death risk
        new_input_death=np.array([[[sex,age,height,weight,bmi,blood_type,insurance,income,race,immigrant,smoking,alcohol,contacts_count,housecount,public_transport,working,worried,self_reducing_risk,self_social_distancing,self_washing_hands,house_reducing_risk,house_social_distancing,house_washing_hands,sanitizer,masks,symptoms,contacts_infected,asthma,kidney_disease,liver_disease,compromised_immune,heart_disease,lung_disease,diabetes,hiv_positive,hypertension,other_chronic,nursing_home,health_worker]]])
        new_input_death1=new_input_death.reshape(1,-1)
        new_output_death= regressor2.predict(new_input_death1)
        print("\nRisk of dying from covid-19", abs(new_output_death),"%")

        freqs5 = {
        'death': abs(new_output_death)[0],
        'risk': abs(new_output_covid)[0],
        "Accuracy for covid risk training set":regressor1.score(X_train1,y_train1)*100,
        "Accuracy for covid risk test set":regressor1.score(X_test1,y_test1)*100,
        "Accuracy for death risk training set":regressor2.score(X_train2,y_train2)*100,
        "Accuracy for death risk test set":regressor2.score(X_test2,y_test2)*100
        }
        return  jsonify(freqs5)


@app.route("/")                   # at the end point /
def hello1():
        
        # bmi= int(request.args.get('bmi'))
        # race=int(request.args.get('race'))
        # immigrant=int(request.args.get('img'))
        # worried=int(request.args.get('wrd'))
        # self_reducing_risk=int(request.args.get('sredris'))
        # house_reducing_risk=int(request.args.get('hredris'))
        freqs1 = {
        # 'testriskurl': 'https://risk-death-covid19-api.herokuapp.com/risk?gen=0&age=18&hgt=123&wgt=34&inc=1&smk=1&alc=3&concnt=0&hcnt=4&wrkng=1&masks=3&sym=2&coninf=0&asthma=0&lung=0&hlth=1',
        # 'testdeathurl':'https://risk-death-covid19-api.herokuapp.com/death?gen=0&age=40&hgt=132&wgt=40&inc=4&smk=1&alc=12&bt=8&ins=1&concnt=20&hcnt=4&ptrnspt=5&wrkng=3&ssocdis=1&swashnds=1&hsocdis=1&hwashnds=1&san=3&masks=2&sym=1&coninf=0&asthma=0&kidney=0&liver=0&cmpimm=0&heart=1&lung=0&diabetes=0&hiv=0&hyprt=1&otherchr=0&nrsng=0&hwork=0',
        'testdeathandriskurl':'https://risk-death-covid19-api.herokuapp.com/deathandrisk?gen=0&age=40&hgt=132&wgt=40&inc=4&smk=1&alc=12&bt=8&ins=1&concnt=20&hcnt=4&ptrnspt=5&wrkng=3&ssocdis=1&swashnds=1&hsocdis=1&hwashnds=1&san=3&masks=2&sym=1&coninf=0&asthma=0&kidney=0&liver=0&cmpimm=0&heart=1&lung=0&diabetes=0&hiv=0&hyprt=1&otherchr=0&nrsng=0&hwork=0&bmi=78&race=1&img=0&wrd=0&sredris=0&hredris=0',
        'options':['search'],
        'riskparams':'gender=gender,age=age,height=hgt,weight=wgt,income=inc,smoking=smk,alcohol_consumed=alc,contacts_count=concnt,housecount=hcnt,working=wrkng,masks=masks,symptoms=sym,contactsinfected=coninf,asthma=asthma,lung=lung,health_worker=hlth',
        }
        return  jsonify(freqs1) 

# @app.route("/riskold")                   # at the end point /
# def hello2():
        
#         gender=int(request.args.get('gen'))
#         age=int(request.args.get('age'))
#         height=int(request.args.get('hgt'))
#         weight=int(request.args.get('wgt'))
#         income=int(request.args.get('inc'))
#         smoking=int(request.args.get('smk'))
#         alcohol=int(request.args.get('alc'))
#         contacts_count = int(request.args.get('concnt'))
#         housecount = int(request.args.get('hcnt'))
#         working=int(request.args.get('wrkng'))
#         masks=int(request.args.get('masks'))
#         symptoms=int(request.args.get('sym'))
#         contactsinfected=int(request.args.get('coninf'))
#         asthma=int(request.args.get('asthma'))
#         lung=int(request.args.get('lung'))
#         health_worker=int(request.args.get('hlth'))

#         new_input_covid=np.array([[gender,age,height,weight,income,smoking,alcohol,contacts_count,housecount,working,masks,symptoms,contactsinfected,asthma,lung,health_worker]])
#         new_input_covid1=new_input_covid.reshape(1,-1)
#         new_output_covid= regressor1.predict(new_input_covid1)
#         print("\nRisk of getting covid-19", abs(new_output_covid),"%")
#         freqs2 = {
#         'predictedoutput': abs(new_output_covid[0])
#         }
#         return  jsonify(freqs2) 


# @app.route("/deathandriskold")                   # at the end point /
# def deathrisk():
        
    
#         gender = int(request.args.get('gen'))
#         age = int(request.args.get('age'))
#         height = int(request.args.get('hgt'))
#         weight = int(request.args.get('wgt'))
#         income = int(request.args.get('inc'))
#         smoking = int(request.args.get('smk'))
#         alcohol = int(request.args.get('alc'))
#         blood_type = int(request.args.get('bt'))
#         insurance = int(request.args.get('ins'))
#         contacts_count = int(request.args.get('concnt'))
#         housecount = int(request.args.get('hcnt'))
#         public_transport = int(request.args.get('ptrnspt'))
#         working = int(request.args.get('wrkng'))
#         self_social_distancing = int(request.args.get('ssocdis'))
#         self_washing_hands = int(request.args.get('swashnds'))
#         house_social_distancing = int(request.args.get('hsocdis'))
#         house_washing_hands = int(request.args.get('hwashnds'))
#         sanitizer = int(request.args.get('san'))
#         masks = int(request.args.get('masks'))
#         symptoms = int(request.args.get('sym'))
#         contactsinfected = int(request.args.get('coninf'))
#         asthma = int(request.args.get('asthma'))
#         kidney_disease = int(request.args.get('kidney'))
#         liver_disease = int(request.args.get('liver'))
#         compromised_immune = int(request.args.get('cmpimm'))
#         heart_disease = int(request.args.get('heart'))
#         lung = int(request.args.get('lung'))
#         diabetes = int(request.args.get('diabetes'))
#         hiv_positive = int(request.args.get('hiv'))
#         hypertension = int(request.args.get('hyprt'))
#         other_chronic = int(request.args.get('otherchr'))
#         nursing_home = int(request.args.get('nrsng'))
#         health_worker = int(request.args.get('hwork'))

#         # risk
#         new_input_covid=np.array([[gender,age,height,weight,income,smoking,alcohol,contacts_count,housecount,working,masks,symptoms,contactsinfected,asthma,lung,health_worker]])
#         new_input_covid1=new_input_covid.reshape(1,-1)
#         new_output_covid= regressor1.predict(new_input_covid1)
#         print("\nRisk of getting covid-19", abs(new_output_covid),"%")

#         # death
#         new_input_death=np.array([[[gender,age,height,weight,income,smoking,alcohol,blood_type,insurance,contacts_count,housecount,public_transport,working,self_social_distancing,self_washing_hands,house_social_distancing,house_washing_hands,sanitizer,masks,symptoms,contactsinfected,asthma,kidney_disease,liver_disease,compromised_immune,heart_disease,lung,diabetes,hiv_positive,hypertension,other_chronic,nursing_home,health_worker]]])
#         new_input_death1=new_input_death.reshape(1,-1)
#         new_output_death= regressor2.predict(new_input_death1)
#         print("\nRisk of dying from covid-19", abs(new_output_death),"%")

#         freqs4 = {
#         'death': abs(new_output_death)[0],
#         'risk': abs(new_output_covid)[0],
#         "Accuracy for covid risk training set":regressor1.score(X_train1,y_train1)*100,
#         "Accuracy for covid risk test set":regressor1.score(X_test1,y_test1)*100,
#         "Accuracy for death risk training set":regressor2.score(X_train2,y_train2)*100,
#         "Accuracy for death risk test set":regressor2.score(X_test2,y_test2)*100
#         }
#         return  jsonify(freqs4)



# @app.route("/deathold")                   # at the end point /
# def death():
        
    
#         gender = int(request.args.get('gen'))
#         age = int(request.args.get('age'))
#         height = int(request.args.get('hgt'))
#         weight = int(request.args.get('wgt'))
#         income = int(request.args.get('inc'))
#         smoking = int(request.args.get('smk'))
#         alcohol = int(request.args.get('alc'))
#         blood_type = int(request.args.get('bt'))
#         insurance = int(request.args.get('ins'))
#         contacts_count = int(request.args.get('concnt'))
#         housecount = int(request.args.get('hcnt'))
#         public_transport = int(request.args.get('ptrnspt'))
#         working = int(request.args.get('wrkng'))
#         self_social_distancing = int(request.args.get('ssocdis'))
#         self_washing_hands = int(request.args.get('swashnds'))
#         house_social_distancing = int(request.args.get('hsocdis'))
#         house_washing_hands = int(request.args.get('hwashnds'))
#         sanitizer = int(request.args.get('san'))
#         masks = int(request.args.get('masks'))
#         symptoms = int(request.args.get('sym'))
#         contactsinfected = int(request.args.get('coninf'))
#         asthma = int(request.args.get('asthma'))
#         kidney_disease = int(request.args.get('kidney'))
#         liver_disease = int(request.args.get('liver'))
#         compromised_immune = int(request.args.get('cmpimm'))
#         heart_disease = int(request.args.get('heart'))
#         lung = int(request.args.get('lung'))
#         diabetes = int(request.args.get('diabetes'))
#         hiv_positive = int(request.args.get('hiv'))
#         hypertension = int(request.args.get('hyprt'))
#         other_chronic = int(request.args.get('otherchr'))
#         nursing_home = int(request.args.get('nrsng'))
#         health_worker = int(request.args.get('hwork'))

#         # death
#         new_input_death=np.array([[[gender,age,height,weight,income,smoking,alcohol,blood_type,insurance,contacts_count,housecount,public_transport,working,self_social_distancing,self_washing_hands,house_social_distancing,house_washing_hands,sanitizer,masks,symptoms,contactsinfected,asthma,kidney_disease,liver_disease,compromised_immune,heart_disease,lung,diabetes,hiv_positive,hypertension,other_chronic,nursing_home,health_worker]]])
#         new_input_death1=new_input_death.reshape(1,-1)
#         new_output_death= regressor2.predict(new_input_death1)
#         print("\nRisk of dying from covid-19", abs(new_output_death),"%")

        freqs3 = {
        'predictedoutput': abs(new_output_death)[0],
        }
        return  jsonify(freqs3)

# @app.after_request
# def after_request(response):
#   response.headers.add('Access-Control-Allow-Origin', '*')
#   response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#   response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#   return response



if __name__ == "__main__":        # on running python app.py
    app.run()                     # run the flask app