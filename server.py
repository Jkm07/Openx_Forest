from flask import Flask, Blueprint, request, render_template
import joblib
import numpy as np
import pandas as ps
import MyHeurstic as mh
import tensorflow as tf


labels = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]

#Nasz strategia
heuristic = mh.my_heuristic

#Model SVC
svc = joblib.load("ModelSVC.sav")

#Model Linear Discriminant
lin = joblib.load("ModelLinearDiscriminant.sav")

#Model Tensor
tensor = tf.keras.models.load_model('ModelNN/ModelNN')

ml = Blueprint('ml', __name__)

def getLabel(x):
    return "{0}({1})".format(labels[x[0] - 1], x[0])

def getSolidType(x):
    out = [0] * 40
    out[x - 1] = 1
    return out

def getWildnesArea(x):
    out = [0] * 4
    out[x - 1] = 1
    return out

def prepareData(int_features):
    wildnesArea = getWildnesArea(int_features[-2])
    solidType = getSolidType(int_features[-1])
    int_features.pop()
    int_features.pop()
    x = np.concatenate([int_features, wildnesArea, solidType])
    return ps.DataFrame([x])

@ml.route('/')
def home():
    return render_template('index.html')

@ml.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    x = prepareData(int_features)

    h_predict = heuristic.predict(x)
    ld_predict = lin.predict(x)
    svc_predict = svc.predict(x)
    tensor_predict = np.argmax(tensor.predict(x), 1) + 1

    return render_template('index.html',
                           heuristic_text='Heuristic: <b>{}</b>'.format(getLabel(h_predict)),
                           ld_text='Linear Discriminant: <b>{}</b>'.format(getLabel(ld_predict)),
                           svc_text='SVC: <b>{}</b>'.format(getLabel(svc_predict)),
                           tensor_text='Tensor: <b>{}</b>'.format(getLabel(tensor_predict)))

if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(ml, url_prefix='/')

    app.run(host="0.0.0.0", port=8080)