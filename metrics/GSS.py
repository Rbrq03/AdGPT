import torch
from PIL import Image
import numpy as np
from diffusers import AutoPipelineForText2Image
from transformers import CLIPImageProcessor, CLIPModel

import torch.nn.functional as F

#Generative Similarity Score
class GSS:

    def __init__(self, SD_model_ID, clip_model_ID, num_inference_steps, devices):
        self.pipe = AutoPipelineForText2Image.from_pretrained(SD_model_ID, torch_dtype=torch.float16, variant="fp16")
        self.pipe = self.pipe.to(devices)

        self.clip_model = CLIPModel.from_pretrained(clip_model_ID)
        self.clip_model = self.clip_model.to(devices)
        self.preprocess = CLIPImageProcessor.from_pretrained(clip_model_ID)

        self.devices = devices
        self.num_inference_steps = num_inference_steps

    def get_score(self, prompt, image_path) -> float:

        #Stage1: generate image
        gr_image = self.pipe(prompt=prompt, num_inference_steps=self.num_inference_steps).images[0]
        gt_image = self.load_and_preprocess_image(image_path)

        gr_image = self.preprocess(gr_image, return_tensors="pt")["pixel_values"].to(self.devices)
        gt_image = gt_image.to(self.devices)

        #Stage2: image embedding
        with torch.no_grad():
            embedding_gt = self.clip_model.get_image_features(gt_image)
            embedding_gr = self.clip_model.get_image_features(gr_image)

        #Stage3: calculate similarity score
        similarity_score = F.cosine_similarity(embedding_gr, embedding_gt)

        return similarity_score.cpu().numpy()[0]
        

    def load_and_preprocess_image(self, image_path):
        image = Image.open(image_path)
        image = self.preprocess(image, return_tensors="pt")["pixel_values"]
        return image