import mlflow
from fastapi import FastAPI

app = FastAPI()

@app.get('/predict')
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float) -> int:
    model = mlflow.sklearn.load_model("mlartifacts/615903767243266673/models/m-961e36c4ed2b42459bfbe8f54c0369e7/artifacts")
    res = model.predict([[5.1, 3.5, 1.4, 0.2]])
    return res
