from flask import Flask,render_template,request,url_for
from transform import data
from model import Data_Model
import numpy as np

app = Flask(__name__)



@app.route("/", methods=['POST', 'GET'])
def index():
    outputData = {}
    arr = None  # Initialize prediction result
    
    if request.method == 'POST':
        # Get the form data
        Pclass = request.form.get('Pclass')
        age = request.form.get('age')
        sex = request.form.get('sex')
        title = request.form.get('Title')
        family_size = request.form.get('Family_size')
        embarked = request.form.get('Embarked')
        fare = request.form.get('fare')
        # print("Before update : ",outputData)
        # Check if the necessary fields are filled
        # print(f"Pclass: {Pclass}, Age: {age}, Sex: {sex}, Title: {title}, Family Size: {family_size}, Embarked: {embarked}, Fare: {fare}")

        if all([Pclass, age, sex, title, family_size, embarked, fare]):
            outputData.update({
                'pclass': [int(Pclass)],
                'age': [int(age)],
                'sex': [sex],
                'title': [title],
                'family_size': [family_size],
                'embarked': [embarked],
                'fare': [float(fare)]
            })
            
            # Perform transformations and model prediction
            transformed = Data_Model(outputData)
            result = transformed.columnTransform()
            LR_pred = transformed.LRmodel(result)
            RFC_pred = transformed.RFCmodel(result)
            SVC_pred = transformed.SVCmodel(result)
            arr  = {"Linear Regression predict ":LR_pred[0],
              "Random Forest Classifier predict ":RFC_pred[0],
              "Support vector machine predict ":SVC_pred[0]
              }
            # print(arr[0])
    # Render the home template, passing both the form data and prediction result
    return render_template("index.html", DataSet=data, output_data=arr)




if __name__ == "__main__":
    app.run(debug=True)

