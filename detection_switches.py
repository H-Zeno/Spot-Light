import numpy as np
import cv2
import supervision as sv
from tqdm import tqdm
from inference.models.yolo_world.yolo_world import YOLOWorld
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import torch
torch.cuda.empty_cache()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

#
# classes = ["light switch", "light switch", "light switch"]

classes = ["round light switch",
           "white light switch",
           "small light switch",
           "push button light switch",
           "rotating light switch",
           "turning round  light switch",
           "double push button light switch",
           "black light switch",
           "knob",
           "car",
           "horse"]

path = "/home/cvg-robotics/tim_ws/YOLO-World/data/switches/IMG_0362 2.jpeg"

model = YOLOWorld(model_id="yolo_world/l")
model.set_classes(classes)

image = cv2.imread("/home/cvg-robotics/tim_ws/YOLO-World/data/switches/IMG_0362 2.jpeg")
results = model.infer(image, confidence=0.005)
detections = sv.Detections.from_inference(results).with_nms(threshold=0.05, class_agnostic=True)

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=1, text_scale=1, text_color=sv.Color.BLACK)

labels = [
    f"{classes[class_id]} {confidence:0.3f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
sv.plot_image(annotated_image, (20, 20))


# second iteration
Bbox = detections.xyxy


switch_bbox = Bbox[1,:]
x1, y1, x2, y2 = switch_bbox.astype(int)
x, y, w, h = x1, y1, x2 - x1, y2 - y1

# Crop the image using NumPy array slicing
cropped_image = image[y:y + h, x:x + w]
sv.plot_image(cropped_image, (20, 20))

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")

# prepare image and text prompt, using the appropriate prompt template
# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)
prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nIs this a push button light switch, OR a rotating button light switch<|im_end|><|im_start|>assistant\n"

affordance_classes = {0: "PUSH",
                      1: "ROTATING",
                      2: "something else"}

def compute_affordance_VLM(cropped_image, affordance_classes, model, processor):
    # build prompt
    prmpt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nIs this "

    for key, value in affordance_classes.items():
        if key < len(affordance_classes)-1:
            prmpt += f"a {value} button light switch (if yes answer {key}) OR "
        else:
            prmpt += f"{value} which is unlikely (if yes answer {key})."
    prmpt += "<|im_end|><|im_start|>assistant\n"

    inputs = processor(prmpt, cropped_image, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=100)

    affordance_key = int(processor.decode(output[0], skip_special_tokens=True)[-2])

    return affordance_classes[0]

affordance = compute_affordance_VLM(cropped_image=cropped_image, affordance_classes=affordance_classes, model=model, processor=processor)

# inputs = processor(prompt, cropped_image, return_tensors="pt").to("cuda:0")
# # autoregressively complete prompt
# output = model.generate(**inputs, max_new_tokens=100)
# #
#
# ans_= processor.decode(output_[0], skip_special_tokens=True)


# model = YOLO("yolov8l-worldv2.pt")
# model.set_classes(classes)
# model.confidence = 0.005
# model.iou_threshold = 0.5
#
# bounding_box_annotator = sv.BoundingBoxAnnotator()
# label_annotator = sv.LabelAnnotator()
#
# test_image = cv2.imread(path)
#
# result = model.predict(test_image)
#
# detections = sv.Detections.from_ultralytics(result[0])
#
# annotated_image = bounding_box_annotator.annotate(
#     scene=test_image.copy(),
#     detections=detections
# )
#
# annotated_image = label_annotator.annotate(
#     scene=annotated_image,
#     detections=detections
# )
#
# sv.plot_image(annotated_image, (20, 20))


# from gradio_client import Client, file
#
#
#
# client = Client("https://stevengrove-yolo-world.hf.space/--replicas/9i6p7/")
# result = client.predict(
# 		file(path),	# filepath  in 'input image' Image component
# 		classes,	# str  in 'Enter the classes to be detected, separated by comma' Textbox component
# 		15,	# float (numeric value between 1 and 300) in 'Maximum Number Boxes' Slider component
# 		0.005,# float (numeric value between 0 and 1) in 'Score Threshold' Slider component
# 		0.5,	# float (numeric value between 0 and 1) in 'NMS Threshold' Slider component
# 		api_name="/partial"
# )
# print(result)
# image_client = cv2.imread(result)
# sv.plot_image(image_client, (20, 20))

a =2