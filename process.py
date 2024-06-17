"""
This code demonstrate how to
- Share medical imagery file
- Query medical imagery file
- Use medical imagery file to train ai model
"""


import logging
import time
import requests

import os
import json

import duckdb
import urllib.request 

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
    logger.info(f"|    https://github.com/./zna_anotations.csv |")
    logger.info(f"|    https://github.com/./zna_dicom.csv |")
    dataProvider1URL="https://github.com/datavillage-me/cage-process-medical-imagery/raw/main/data/zna_anotations.csv"
    #dataProvider1URL="data/zna_anotations.csv"
    dataProvider2URL="https://github.com/datavillage-me/cage-process-medical-imagery/raw/main/data/zna_dicom.csv"
    #dataProvider2URL="data/zna_dicom.csv"
    start_time = time.time()
    logger.info(f"|    Start time:  {start_time} secs |")
    parameters=evt.get("parameters", "")
    whereClause=parameters["id"]
    
    if whereClause!='':
        baseQuery="SELECT path from '"+dataProvider1URL+"' as zna_anotations,'"+dataProvider2URL+"' as images WHERE zna_anotations.Studienummer=images.id AND images.id='"+whereClause+"'"
    else:
        baseQuery="SELECT path from '"+dataProvider1URL+"' as zna_anotations,'"+dataProvider2URL+"' as images WHERE zna_anotations.Studienummer=images.id"
    
    df=duckdb.sql(baseQuery).df()
    execution_time=(time.time() - start_time)
    logger.info(f"|    Execution time:  {execution_time} secs |")

    patologist=parameters["pathologist"]
    logger.info(f"| 2. Share outputs to pathologist(s): {patologist}      |")
    
    # dicom_to_share = df['path'][0]
    # dicom_as_bytes = open(dicom_to_share, "rb") 
    # with open("my.dcm", 'wb') as binary_file:
    #    binary_file.write(dicom_as_bytes.read())

    logger.info(f"| Creation of unique links (form & image)        |")
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

    logger.info(f"| Creation of unique links (form & image)        |")
    logger.info(f"| Send notification to patologist(s): {patologist} |")

    #send notification to pathologist 
    url = "https://script.google.com/macros/s/AKfycbxDH00o1yDRty5W3yHfdZMgJaKxAcPS4VDignS_8EHH2IrNJEGvxY4X8AD2FYtJSjxFRQ/exec"

    payload = json.dumps({
    "sender": f"{patologist}",
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
   

if __name__ == "__main__":
    test_event = {
            "type": "SHARE",
            "parameters": 
            {
                "id":"s00001",
                "pathologist":"45920239"
            }
    }
    process_share_event(test_event)