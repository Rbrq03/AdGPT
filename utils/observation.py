import easyocr
import cnocr


def get_ocr(image, version="en"):

    if version == "en":
        reader = easyocr.Reader(["en"], detect_network="dbnet18", gpu=True)
        result = reader.readtext(image, detail=0, paragraph=True)
    elif version == "cn":
        reader = cnocr.CnOcr(context="cuda:0")
        result = reader.ocr(image)
        result = ";".join([item["text"] for item in result])
    else:
        raise NotImplementedError("This version of OCR model is not implemented.")

    return result


def get_caption(image, processor, model):

    text = "a picture of"
    inputs = processor(image, text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
