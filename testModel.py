import json

from Hand_Detector import Recorder
import numpy as np
import requests
import json

def send_features_to_server(feature_vector):
    feature_vector = np.array(feature_vector).flatten()

    url = "http://127.0.0.1:5000/predict"

    payload = {
        "sequence": feature_vector.tolist()
    }
    try:
        response = requests.post(url, json=payload)
    except:
        return {"gesture": "error"}
    if response.status_code == 200:
        return response.json()
    else:
        return {"gesture": "error"}




rec = Recorder()
print('jdsng')
while True:
    rec.checkMotion()
    # print(rec.data)
    if rec.data:
        data = send_features_to_server(rec.data)
        print(data)
        rec.resetData()
    # rec.destroyWindow()