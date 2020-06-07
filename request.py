# -*- Rush Alemu @ SIS -*-
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'Age': 40, 'Gender': 1, 'Comorbid': 0, 'CurrentSmoker': 0, 'RespiratoryRateGreaterThan24': 0, 'TemperatureGreaterThan37': 0, 'GroundGlassOpacity': 0,
                             'WBC': 5, 'LymphocyteCount': 0.6, 'Platelets': 100, 'Albumin': 30, 'LactateDehydrogenase': 200, 
     'TroponinI': 3, 'Ddimer': 0.6, 'Ferritin': 500, 'Interleukin6': 5, 'Procalcitonin': 0.1})

print(r.json())
