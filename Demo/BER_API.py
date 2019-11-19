from flask import Flask, request,  abort, jsonify, request
import pandas as pd
import numpy as np
import json
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/pred',methods=['POST','GET'])
def make_prediction():
    text = json.dumps(request.get_json(force=True))
    text = json.loads(text)['0']
    
    
    file = open('sample.txt', 'w') 
    file.write(text)  
    file.close()
    
    myCmd = 'java -cp ../code/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ../code/bio-ner-model2.ser.gz -textFile sample.txt -outputFormat tsv > prediction.tsv'
    os.system(myCmd)
    
    df = pd.read_csv('prediction.tsv', sep = '\t', header= None)
    df.columns = ['Token', 'Tag']
    return df.to_json(orient='records')

@app.route('/hello')
def test():
    return 'Hello back'

if __name__ == '__main__':
    app.run(port=3231, debug = True)