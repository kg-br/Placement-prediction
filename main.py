from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict(final)
    output=prediction[0]
    if output == 1:
        return render_template('index.html',pred='You have higher chances of getting Placed')
    else:
        return render_template('index.html',pred='Sorry, You have higher chances of not getting placed')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)