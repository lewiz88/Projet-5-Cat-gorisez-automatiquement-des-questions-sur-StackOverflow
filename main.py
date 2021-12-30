
from fastapi import FastAPI
from modules import predicted_tags, post_preprocessing
from pydantic import BaseModel

app = FastAPI(title='API pour prédiction de tags sur Stack Overflow',
              description='Elaboration de Tags pertinents à une question sur Stack Overflow ',
              version='1.0.0')

@app.get("/")
def root():
    return {"Bienvenue sur l'API de prédiction de Tags"}

class Post(BaseModel):
    text : str

@app.post("/prediction")
def get_prediction(text: Post):
    a = post_preprocessing(text.text)
    tags = predicted_tags(a)
    return {'tags': list(tags) }