import numpy as np
from flask import Flask,render_template,request
from keras.models import load_model
import requests

API_KEY = "Rg0cfF6TtFNPcwNKGgF_2IN9wC5hvgnfHA0ZmzykQ2cK"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app = Flask(__name__)
model = load_model('crude_oil.h5',)

@app.route('/')
def home() :
    return render_template("file1.html")
@app.route('/about')
def home1() :
    return render_template("file1.html")
@app.route('/predict')
def home2() :
    return render_template("file2.html")

@app.route('/login',methods = ['POST'])
def login() :
    
    a=request.form['year1']
    b=request.form['year2']
    c=request.form['year3']
    d=request.form['year4']
    e=request.form['year5']
    f=request.form['year6']
    g=request.form['year7']
    h=request.form['year8']
    i=request.form['year9']
    j=request.form['year10']
    x_input = [a,b,c,d,e,f,g,h,i,j]
    for i in range(0, len(x_input)): 
        x_input[i] = float(x_input[i]) 
    print(x_input)
    x_input=np.array(x_input).reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    n_steps=10
    i=0
    while(i<1):
        
        if(len(temp_input)>10):
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1

    print(lst_output)
    
    
    return render_template("file2.html",showcase = 'The next day predicted value is:'+str(lst_output))
payload_scoring = {"input_data": [{"field": [[['a'],['b'],['c'],['d'],['e'],['f'],['g'],['h'],['i'],['j']]], "values": [[[73.93],[73.78],[73.05],[74.19],[73.89],[74.13],[73.45],[77.41],[75.23],[69.91]]]}]}

response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/3c43fb86-9f0b-4f09-a5be-3b7756eda643/predictions?version=2022-11-15', json=payload_scoring,
 headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
predictions=response_scoring.json()
pred=predictions['predictions'][0]['values'][0][0]
print(pred)
if __name__ == '__main__' :
    app.run(debug = True,port=5000)