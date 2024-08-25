import cv2
import sys
import torch
import argparse

from utils.llmchat import LLMs_chat
from utils.util import cv2pil, pil2cv, crop_img, check_args
from utils.observation import get_caption, get_ocr

import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

sys.path.insert(0, "third_party/CenterNet2/")
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test

from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
from transformers import AutoProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import BlipProcessor


HINT_PROMPT = open("./prompts/hint.txt", "r").read()
CLASSIFY_ROLE = open("./prompts/classify.txt", "r").read()
ROLE_PROMPT = open("./prompts/role.txt", "r").read()


class AdGPT:

    def __init__(self, args):

        self.predictor, self.metadata = self.set_detic()
        self.processor, self.model = self.set_caption_model()
        self.version = args.version

        if self.openai:
            self.api_key = args.openai_key
            self.base_url = args.openai_base
            self.chat_model = args.openai_model
            self.llm = "openai"
        elif self.chatglm:
            self.api_key = args.glm_key
            self.chat_model = args.openai_model
            self.base_url = None
            self.llm = "glm"

    def set_detic(self):

        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(
            "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
        )
        cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = (
            True  # For better visualization purpose. Set to False for all classes.
        )
        cfg.FP16 = False

        predictor = DefaultPredictor(cfg)

        BUILDIN_CLASSIFIER = {
            "lvis": "datasets/metadata/lvis_v1_clip_a+cname.npy",
            "objects365": "datasets/metadata/o365_clip_a+cnamefix.npy",
            "openimages": "datasets/metadata/oid_clip_a+cname.npy",
            "coco": "datasets/metadata/coco_clip_a+cname.npy",
        }

        BUILDIN_METADATA_PATH = {
            "lvis": "lvis_v1_val",
            "objects365": "objects365_v2_val",
            "openimages": "oid_val_expanded",
            "coco": "coco_2017_val",
        }

        vocabulary = "lvis"
        metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
        classifier = BUILDIN_CLASSIFIER[vocabulary]
        num_classes = len(metadata.thing_classes)
        reset_cls_test(predictor.model, classifier, num_classes)

        return predictor, metadata

    def set_caption_model(self, model_id="Salesforce/blip-image-captioning-large"):
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id).to("cuda")

        return processor, model

    def get_observation(self):

        img = cv2.imread(self.img_path)
        image = cv2pil(img)

        info_per_obj = []
        pred_obj = self.predictor(img)

        classes = [
            self.metadata.thing_classes[x]
            for x in pred_obj["instances"].pred_classes.cpu().tolist()
        ]
        boxes = [i.cpu().numpy() for i in pred_obj["instances"].pred_boxes]

        for i in range(len(classes)):
            obj_crop = crop_img(img, boxes[i])
            obj_crop = cv2pil(obj_crop)
            obj_caption = get_caption(obj_crop, self.processor, self.model)
            info_per_obj.append({classes[i]: obj_caption})

        observation = {
            "Image_caption": get_caption(image, self.processor, self.model),
            "OCR": get_ocr(img, self.version),
            "Info_per_object": info_per_obj,
        }

        return observation

    def classify(self):

        messages = [
            {"role": "system", "content": CLASSIFY_ROLE},
        ]

        response = LLMs_chat(
            messages, self.api_key, self.base_url, self.chat_model, self.llm
        )

        messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        info = HINT_PROMPT + "\n" + str(self.observation)
        messages.append({"role": "user", "content": info})

        response = LLMs_chat(
            messages, self.api_key, self.base_url, self.chat_model, self.llm
        )

        return response

    def predict(self, image_path):
        self.img_path = image_path
        self.observation = self.get_observation()

        classify_res = self.classify()

        messages = [
            {"role": "system", "content": ROLE_PROMPT},
            {
                "role": "user",
                "content": f"You need to summarize a {classify_res} ad, first you only need to generate the general chain of reasoning of ad with type {classify_res} based on the characteristics of the {classify_res} ad, using A, B, C... to show the inference chain order",
            },
        ]

        response = LLMs_chat(
            messages, self.api_key, self.base_url, self.chat_model, self.llm
        )

        prompt1 = "Analyze the content of this ad step by step according to the chain of reasoning given above, and then summarize the ad for me. The result of the analysis starts with Thought:, and the final result of the summary starts with Summary:"
        prompt2 = "Here is visual result:" + str(self.observation)

        messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )
        messages.append({"role": "user", "content": prompt1})
        messages.append({"role": "user", "content": prompt2})

        response = LLMs_chat(
            messages, self.api_key, self.base_url, self.chat_model, self.llm
        )
        index = response.find("Summary")
        response = response[index:]

        return messages, response


def get_args():
    parser = argparse.ArgumentParser(description="AdGPT")
    parser.add_argument("--image_path", type=str, help="Path to the image file")
    parser.add_argument(
        "--version", type=str, help="AdGPT version, currently support cn and en"
    )

    # Openai Argument
    parser.add_argument("--openai", action="store_true", help="use openai series model")
    parser.add_argument("--openai_key", type=str, help="openai api_key", default=None)
    parser.add_argument("--openai_base", type=str, help="openai api_base", default=None)
    parser.add_argument(
        "--openai_model", type=str, help="openai chat model name", default="gpt-4o-mini"
    )

    # GLM Argument
    parser.add_argument("--chatglm", action="store_true", help="use glm serise model")
    parser.add_argument("--glm_key", type=str, help="glm api_key", default=None)
    parser.add_argument(
        "--glm_model",
        type=str,
        help="glm chat model name",
        default="glm4",
    )

    args = parser.parse_args()

    check_args(args)

    return args


if __name__ == "__main__":

    args = get_args()
    agent = AdGPT(args)
    _, result = agent.predict(args.image_path)
    print(result)
