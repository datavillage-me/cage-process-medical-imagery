"""
This code demonstrate how to
- Share medical imagery file
- Query medical imagery files
- Use medical imagery file to predict cancer - https://www.kaggle.com/code/subhajeetdas/brain-tumor-cancer-detection-with-cnn-100-acc
"""


import logging
import time
import requests

import os
import json

import duckdb
import urllib.request 
from tensorflow.keras.models import Sequential, load_model,model_from_json
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np
from PIL import Image

from dv_utils import default_settings, Client 

import pandas as pd

logger = logging.getLogger(__name__)

input_dir = "/resources/data"
output_dir = "/resources/outputs"

# let the log go to stdout, as it will be captured by the cage operator
logging.basicConfig(
    level=default_settings.log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# define an event processing function
def event_processor(evt: dict):
    """
    Process an incoming event
    Exception raised by this function are handled by the default event listener and reported in the logs.
    """
    
    logger.info(f"Processing event {evt}")

    # dispatch events according to their type
    evt_type =evt.get("type", "")
    if(evt_type == "SHARE"):
        process_share_event(evt)
    elif (evt_type == "QUERY"):
        process_query_event(evt)
    elif (evt_type == "INFER"):
        process_infer_event(evt)
    else:
        generic_event_processor(evt)


def generic_event_processor(evt: dict):
    # push an audit log to reccord for an event that is not understood
    logger.info(f"Received an unhandled event {evt}")

def process_share_event(evt: dict):
    logger.info(f"--------------------------------------------------")
    logger.info(f"|               START SHARING                    |")
    logger.info(f"|                                                |")
    
    # load the image data from data provider
    # duckDB is used to load the data and aggregated them in one single datasets
    logger.info(f"| 1. Load data from data providers               |")
    logger.info(f"|    https://github.com/./zna_anotations.csv     |")
    logger.info(f"|    https://github.com/./zna_dicom.csv          |")
    dataProvider1URL="https://github.com/datavillage-me/cage-process-medical-imagery/raw/main/data/zna_anotations.csv"
    #dataProvider1URL="data/zna_anotations.csv"
    dataProvider2URL="https://github.com/datavillage-me/cage-process-medical-imagery/raw/main/data/zna_dicom.csv"
    #dataProvider2URL="data/zna_dicom.csv"
    start_time = time.time()
    logger.info(f"|    Start time:  {start_time} secs          |")
    parameters=evt.get("parameters", "")
    whereClause=parameters["id"]
    
    if whereClause!='':
        baseQuery="SELECT path from '"+dataProvider1URL+"' as zna_anotations,'"+dataProvider2URL+"' as images WHERE zna_anotations.Studienummer=images.id AND images.id='"+whereClause+"'"
    else:
        baseQuery="SELECT path from '"+dataProvider1URL+"' as zna_anotations,'"+dataProvider2URL+"' as images WHERE zna_anotations.Studienummer=images.id"
    
    df=duckdb.sql(baseQuery).df()
    execution_time=(time.time() - start_time)
    logger.info(f"|    Execution time:  {execution_time} secs        |")

    patologist=parameters["pathologist"]
    logger.info(f"| 2. Share outputs to pathologist(s): {patologist}    |")
    
    # dicom_to_share = df['path'][0]
    # dicom_as_bytes = open(dicom_to_share, "rb") 
    # with open("my.dcm", 'wb') as binary_file:
    #    binary_file.write(dicom_as_bytes.read())

    #write dicom image in output with unique link
    dicom_to_share = "https://github.com/datavillage-me/cage-process-medical-imagery/raw/main/"+df['path'][0]
    with urllib.request.urlopen(dicom_to_share) as f: 
        dicom_as_bytes = f.read()
    
    with open("/resources/outputs/image-"+patologist+"-KFJE340RKDFNZE.dcm", 'wb') as binary_file:
       binary_file.write(dicom_as_bytes)

    #write demand form in output with unique link
    form_to_share = "https://github.com/datavillage-me/cage-process-medical-imagery/raw/main/data/zna_files/form.png"
    with urllib.request.urlopen(form_to_share) as f: 
        form_as_bytes = f.read()
    
    with open("/resources/outputs/form-"+patologist+"-KFJE340RKDFNZE.png", 'wb') as binary_file:
       binary_file.write(form_as_bytes)

    logger.info(f"|    Creation of unique links (form & image)     |")
    logger.info(f"|    Send notification to patologist(s): {patologist}|")

    #send notification to pathologist 
    url = "https://script.google.com/macros/s/AKfycbxDH00o1yDRty5W3yHfdZMgJaKxAcPS4VDignS_8EHH2IrNJEGvxY4X8AD2FYtJSjxFRQ/exec"
    
    sender=parameters["sender"]
    payload = json.dumps({
    "sender": f"{sender}",
    "parameters": {
        "form": "https://api.datavillage.me/collaborationSpaces/jplpngge/cage/resources/output/form-45920239-KFJE340RKDFNZE.png",
        "image": "https://api.datavillage.me/collaborationSpaces/jplpngge/cage/resources/output/image-45920239-KFJE340RKDFNZE.dcm"
    }
    })
    headers = {
    'Content-Type': 'application/json'
    }

    requests.request("POST", url, headers=headers, data=payload)

   
    logger.info(f"|                                                |")
    logger.info(f"--------------------------------------------------")

def process_query_event(evt: dict):
    logger.info(f"--------------------------------------------------")
    logger.info(f"|               START QUERY                      |")
    logger.info(f"|                                                |")
    
    # load the image data from data provider
    # duckDB is used to load the data and aggregated them in one single datasets
    logger.info(f"| 1. Load data from data providers               |")
    logger.info(f"|    https://github.com/./zna_anotations.csv     |")
    logger.info(f"|    https://github.com/./uzgent_anotations.csv          |")
    dataProvider1URL="https://github.com/datavillage-me/cage-process-medical-imagery/raw/main/data/zna_anotations.csv"
    #dataProvider1URL="data/zna_anotations.csv"
    dataProvider2URL="https://github.com/datavillage-me/cage-process-medical-imagery/raw/main/data/uzgent_anotations.csv"
    #dataProvider2URL="data/uzgent_anotations.csv"
    start_time = time.time()
    logger.info(f"|    Start time:  {start_time} secs          |")
    whereClause=evt.get("parameters", "")
    
    if whereClause!='':
        baseQuery="SELECT COUNT(*) as total from read_csv(['"+dataProvider1URL+"','"+dataProvider2URL+"'], union_by_name = true) WHERE "+whereClause
    else:
        baseQuery="SELECT COUNT(*) as total from read_csv(['"+dataProvider1URL+"','"+dataProvider2URL+"'], union_by_name = true)"
    df=duckdb.sql(baseQuery).df()
    totalImages=df['total'][0]
    logger.info(f"|    Nbre of images:  {totalImages}              |")


    #vascular_embolization
    #yes
    df = duckdb.sql(baseQuery+ " AND vascular_embolization=1").df()
    totalVascularEmbolizationYes=df['total'][0]
    #no
    df = duckdb.sql(baseQuery+ " AND vascular_embolization=0").df()
    totalVascularEmbolizationNo=df['total'][0]

    #neoadjuvant_treatment
    #yes
    df = duckdb.sql(baseQuery+ " AND neoadjuvant_treatment=1").df()
    totalNeoadjuvantTreatmentYes=df['total'][0]
    #no
    df = duckdb.sql(baseQuery+ " AND neoadjuvant_treatment=0").df()
    totalNeoadjuvantTreatmentNo=df['total'][0]


    execution_time=(time.time() - start_time)
    logger.info(f"|    Execution time:  {execution_time} secs        |")

    logger.info(f"| 2. Save outputs of the collaboration           |")
    # The output file model is stored in the data folder
    
    output= ''' {
    "images": '''+str(totalImages)+''',
        "vascular_embolization": {
        "yes":'''+str(totalVascularEmbolizationYes)+''',
        "no":'''+str(totalVascularEmbolizationNo)+'''
        },
        "neoadjuvant_treatment": {
        "yes":'''+str(totalNeoadjuvantTreatmentYes)+''',
        "no":'''+str(totalNeoadjuvantTreatmentNo)+'''
        }
    } '''

    #with open('data/my.json', 'w', newline='') as file:
    #    file.write(output)

    with open('/resources/outputs/images-report.json', 'w', newline='') as file:
        file.write(output)
   
    logger.info(f"|                                                |")
    logger.info(f"--------------------------------------------------")

def process_infer_event(evt: dict):
    logger.info(f"--------------------------------------------------")
    logger.info(f"|               START INFERENCE                  |")
    logger.info(f"|                                                |")
    
    # load the image data from data provider
    logger.info(f"| 1. Load data from data providers               |")
    logger.info(f"|    https://github.com/./uzgent_files/          |")
    start_time = time.time()
    logger.info(f"|    Start time:  {start_time} secs          |")
    imageId=evt.get("image_id", "")
    test_image_file=f"/resources/data/{imageId}"
    #test_image_file="data/uzgent_files/g00001.jpg"

    #load AI model
    logger.info(f"| 2. Load AI model                               |")
    #with open('model/model.json', 'r', newline='') as file:
    with open('/resources/data/model.json', 'r', newline='') as file:
        jsonModel = json.load(file)
        best_model = model_from_json(json.dumps(jsonModel))


    #load perform inference
    logger.info(f"| 3. Predict aneurysm - cancer - tumor           |")
    class_probabilities, predicted_class_index = test_image(test_image_file, best_model)

    classes = ["aneurysm", "cancer", "tumor"]
    for i, class_label in enumerate(classes):
        probability = class_probabilities[i]
        logger.info(f'|  Class: {class_label}, Probability: {probability:.4f} |')

    # Calculate and display the predicted class
    predicted_class = classes[predicted_class_index]
    logger.info(f'|  The image is classified as: {predicted_class}|  ')
    
    output= ''' {
        "image_id": "'''+str(imageId)+'''",
        "class": "'''+str(predicted_class)+'''",
        "probabilities":{
        "'''+str(classes[0])+'''":"'''+str(class_probabilities[0])+'''",
        "'''+str(classes[1])+'''":"'''+str(class_probabilities[1])+'''",
        "'''+str(classes[2])+'''":"'''+str(class_probabilities[2])+'''"
        }
    } '''

    #print(output)
    with open('/resources/outputs/score.json', 'w', newline='') as file:
        file.write(output)

    logger.info(f"|                                                |")
    logger.info(f"--------------------------------------------------")

def test_image(file_path, model):
    # Load and preprocess the image
    target_shape = (400, 400)
    image = Image.open(file_path)
    image = image.resize(target_shape)
    img_array = np.asarray(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Get the class probabilities
    class_probabilities = predictions[0]
    
    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)
    
    return class_probabilities, predicted_class_index

if __name__ == "__main__":
    test_event = {
            "type": "SHARE",
            "parameters": 
            {
                "id":"s00001",
                "sender":"34594565",
                "pathologist":"45920239"
            }
    }
    test_event = {"type":"QUERY","parameters":"Tumor_location='Endometrium' AND Histological_subtype='Endometrioid' AND Sample_type='biopsy'"}
    
    test_event = {
        "type": "INFER",
        "image_id": "g00002.jpg"
    }

    
    process_infer_event(test_event)