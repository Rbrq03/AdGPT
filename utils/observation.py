import easyocr

def get_ocr(image):

    reader = easyocr.Reader(["en"], detect_network="dbnet18", gpu=True)
    result = reader.readtext(image, detail=0, paragraph=True)
    return result

def get_caption(image, processor, model):

    text = "a picture of"
    inputs = processor(image, text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)