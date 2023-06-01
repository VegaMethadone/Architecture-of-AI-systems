from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from model_wrapper import get_text_lable

app = FastAPI()


@app.get('/')
def read_root():
    return {'Hello': 'World'}


class LableTextReq(BaseModel):
    text: str 

@app.post('/lable_text')
def read_item(req: LableTextReq):
    text = req.text
    lable = get_text_lable(text)

    return {
        'lable': lable,
        'text_len': len(text),
    }

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")