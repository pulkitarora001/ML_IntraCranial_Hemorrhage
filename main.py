from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import keras
import os


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

brain_stroke_model = keras.models.load_model("resnet-weights (1).hdf5")


@app.post('/brain_stroke_prediction')
def brain_stroke_pred(brain_ct_image: UploadFile):

    if brain_ct_image.size == 0:
        return {"message": "No file was uploaded"}    

    try:
        os.mkdir("brain_ct_images")
    except Exception as e:
        print(e)
    
    file_name=os.getcwd()+"\\brain_ct_images\\"+brain_ct_image.filename.replace(" ","-")

    with open(file_name,'wb+') as f:
        f.write(brain_ct_image.file.read())
        f.close()
    
    img_path = cv2.imread(file_name)

    img = img_path *1./255.
    img = cv2.resize(img, (256,256))
    img = np.array(img, dtype=np.float64)
    img = np.reshape(img, (1,256,256,3))

    is_defect = brain_stroke_model.predict(img)

    if np.argmax(is_defect) == 0:
        status = False
    else:
        status = True

    return {"Tumor_Present": status}

