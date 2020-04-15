
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

gender = {'male':1,'female':0} 
education={'a level or equivalent':0,'he qualification':1,'lower than a level':2,
           'no formal quals':3,'post graduate qualification':4}
disability={'yes':1,'no':0}
age={'0-35':0,'35-55':1,'55-75':2}
region={'east anglian region':0,'wales':10,'scotland':6,'south region':8,'london region':3,
        'west midlands region':11,'south west region':9,'south east region':7,
        'east midlands region':1,'north western region':5,'yorkshire region':12,
        'ireland':2,'north region':4}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    print(int_features)
    int_features[1]=gender[int_features[1]]
    int_features[2]=region[int_features[2]]
    int_features[3]=education[int_features[3]]	
    int_features[4]=age[int_features[4]]
    int_features[7]=disability[int_features[7]]
    print(int_features)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = (prediction[0])
    if output<98:
        output=output
    else:
        output=98

    return render_template('index.html', prediction_text= '{}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)




