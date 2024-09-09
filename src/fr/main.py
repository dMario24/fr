from typing import Union
from fastapi import FastAPI
import pickle
from fr.mmanager import get_model_path

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "fr"}


def run_prediction(length: float):
    model_path = get_model_path()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    r = model.predict([[length**2, length]])
    return float(r[0])

    

@app.get("/fish")
def fish(length: float):
    """
    물고기의 무게 예측

    Args:
        length (float): 물고기 길이(cm)

    Returns:
        dict:
          weight (float): 물고기 무게(g)
          length (float): 물고기 길이(cm)
    """
    fish_weight = run_prediction(length)

    return {
                "length": length, 
                "weight": fish_weight
            }