from fastapi import FastAPI
from pydantic import BaseModel
import pickle 
import pandas as pd


app = FastAPI()


class ScoringTtem(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float


with open('model.pkl','rb') as f:
    model = pickle.load(f)


@app.post('/')
async def scoring_endpoint(item:ScoringTtem):
    df= pd.DataFrame([item.model_dump().values()],columns=['sepal length (cm)',	'sepal width (cm)',	'petal length (cm)',	'petal width (cm)'])
    
    y_hat = model.predict(df).tolist()
    return  {"predict":y_hat}