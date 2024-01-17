from pydantic import BaseModel

class Prediction_Input(BaseModel):
    id : int
    input_data: list

class Prediction_Output(BaseModel):
    id : int
    input_data: list
    pred: int
    pred_label: str
    ground_truth: int
    ground_truth_label: str
    