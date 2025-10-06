# app.py
from fastapi import FastAPI, Request
import fasttext
app = FastAPI(title="Text Checker API")

model = fasttext.load_model("lid.176.bin")# wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
def detect_vietnamese_text(text: str) -> bool:

    lang, prob = model.predict(text)
    lang_code = lang[0].replace("__label__", "")
    confidence = prob[0]

    # nếu không phải tiếng Việt hoặc độ tin cậy thấp
    if lang_code != "vi" or confidence < 0.7:
        return False
    else:
        return True


@app.post("/detect-vietnamese")
async def check_text(request: Request):
    data = await request.json()
    title = data.get("title","")
    overview = data.get("overview","")
    text = title +" "+ overview

    result = detect_vietnamese_text(text)
    print(("Movie " + title + " is "+ "Vietnamese.")  if result else (f"\033[31mMovie " + title + " is not Vietnamese.\033[0m"))
    return {"result": result}
