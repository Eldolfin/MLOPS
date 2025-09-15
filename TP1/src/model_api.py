from fastapi import FastAPI
import joblib
from train_model import model_file

app = FastAPI()
model = joblib.load(model_file)

@app.post("/predict")
def predict(size: float, nb_rooms: int, garden: bool) -> dict:
    input = [[size, nb_rooms, garden]]
    price = model.predict(input)
    return {"y_pred": price[0]}
