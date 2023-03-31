import string
import nltk
from nltk.stem import *
from nltk.corpus import stopwords 
from pydantic import BaseModel
from fastapi import FastAPI
import pickle


def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  ps = PorterStemmer()
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)


model=pickle.load(open('model.pkl','rb'))
tf=pickle.load(open('vectorizer.pkl','rb'))


class Query(BaseModel):
    message: str

app = FastAPI()

@app.get("/")
async def root(query:Query):
    message_transformed=transform_text(query.message)
    message_vector=tf.transform([message_transformed])
    pred = model.predict(message_vector)[0]

    if(pred==1):
      return {"spam": 1}
    else:
      return {"spam": 0}