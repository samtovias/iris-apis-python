from fastapi import APIRouter, HTTPException

from models import Prediction_Input, Prediction_Output
import pickle

router = APIRouter()

""" Load code to predict """
class_labels = ['setosa', 'versicolor', 'virginica']
MODEL_PATH = "model.sav"
model = pickle.load(open(MODEL_PATH,'rb'))

preds = []

@router.get("/ml")
def get_preds():
    return preds

@router.post("/ml", status_code=201, response_model=Prediction_Output)
def predict(pred_input:Prediction_Input):
    prediction_f = model.predict([pred_input.input_data[0][:-1]])
    prediction_dict = {
            "id": pred_input.id,
            "input_data": pred_input.input_data,
            "pred": int(prediction_f),
            "pred_label": class_labels[int(prediction_f)],
            "ground_truth": pred_input.input_data[0][-1],
            "ground_truth_label": class_labels[int(pred_input.input_data[0][-1])]
            }

    preds.append(prediction_dict)

    return prediction_dict

@router.delete("/ml")
def delete_pred(pred_id:int):
    for i in range(len(preds)):
        if preds[i]["id"] == pred_id:
            preds.pop(i)
            return {'Message': "Prediction with id {} deleted!".format(pred_id)}
        
    raise HTTPException(status_code=404, detail=f'Element not found :/') 
        
    
    
