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
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
from skimage.transform import resize

torch.cuda.empty_cache()



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

path = "/home/cvg-robotics/tim_ws/data/switches/img_closeup_3.png"

model = YOLOWorld(model_id="yolo_world/l")
model.set_classes(classes)

def _filter_detections(detections):

    squaredness = (np.minimum(detections.xyxy[:,2] -detections.xyxy[:,0], detections.xyxy[:,3] -detections.xyxy[:,1])/
                   np.maximum(detections.xyxy[:,2] -detections.xyxy[:,0], detections.xyxy[:,3] -detections.xyxy[:,1]))

    idx_dismiss = np.where(squaredness < 0.95)[0]

    filtered_detections = sv.Detections.empty()
    filtered_detections.class_id = np.delete(detections.class_id, idx_dismiss)
    filtered_detections.confidence = np.delete(detections.confidence, idx_dismiss)
    filtered_detections.data['class_name'] = np.delete(detections.data['class_name'], idx_dismiss)
    filtered_detections.xyxy = np.delete(detections.xyxy, idx_dismiss, axis=0)

    return filtered_detections


def _callback(image_slice: np.ndarray) -> sv.Detections:
    result = model.infer(image_slice, confidence=0.005)
    return sv.Detections.from_inference(result).with_nms(threshold=0.05, class_agnostic=True)

image = cv2.imread(path)

slicer = sv.InferenceSlicer(callback = _callback, slice_wh=(image.shape[0]//1, image.shape[1]//1), overlap_ratio_wh=(0.2,0.2))

# results = model.infer(image, confidence=0.005)
# detections = sv.Detections.from_inference(results).with_nms(threshold=0.05, class_agnostic=True)
detections = slicer(image)

detections = _filter_detections(detections=detections)

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


Bbox = detections.xyxy


switch_bbox = Bbox[4,:]
x1, y1, x2, y2 = switch_bbox.astype(int)
x, y, w, h = x1, y1, x2 - x1, y2 - y1

# Crop the image using NumPy array slicing
cropped_image = image[y:y + h, x:x + w]
sv.plot_image(cropped_image, (20, 20))

####################################
# refine button
####################################
# classes = ["button"]
#
# model.set_classes(classes)
# results = model.infer(cropped_image, confidence=0.001)
# detections = sv.Detections.from_inference(results).with_nms(threshold=0.05, class_agnostic=True)
#
# annotated_image = cropped_image.copy()
# annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
# annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
# sv.plot_image(annotated_image, (20, 20))
#
#
# a = 2




processor_llava = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model_llava = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model_llava.to("cuda:0")

affordance_classes = {0: "SINGLE PUSH",
                      1: "DOUBLE PUSH",
                      2: "ROTATING",
                      3: "something else"}


def compute_affordance_VLM_GPT4(cropped_image, affordance_classes):

    # build prompt
    prmpt = "Is this "

    for key, value in affordance_classes.items():
        if key < len(affordance_classes) - 1:
            prmpt += f"a {value} button light switch (if yes answer {key}) OR "
        else:
            prmpt += f"{value} which is unlikely (if yes answer {key})."

    # resize image
    resized_image = cv2.resize(cropped_image, (512, 512))

    # encode image
    image = Image.fromarray(resized_image)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_bytes = buffer.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

<<<<<<< HEAD
    api_key = "..."
=======
    api_key = "..." # fill in api key
>>>>>>> 0bd7598... some experimenting for gpt4
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prmpt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        affordance_key = int(response.json()["choices"][0]["message"]["content"])
    else:
        print("Error:", response.status_code)

    return affordance_key
def compute_affordance_VLM_llava(cropped_image, affordance_classes, model, processor):

    prmpt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nIs this "

    for key, value in affordance_classes.items():
        if key < len(affordance_classes) - 1:
            prmpt += f"a {value} button light switch (if yes answer {key}) OR "
        else:
            prmpt += f"{value} which is unlikely (if yes answer {key})."
    prmpt += "<|im_end|><|im_start|>assistant\n"

    inputs = processor(prmpt, cropped_image, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=100)

    affordance_key = int(processor.decode(output[0], skip_special_tokens=True)[-2])

    return affordance_key


affordance_llava = compute_affordance_VLM_llava(cropped_image=cropped_image, affordance_classes=affordance_classes, model=model_llava, processor=processor_llava)
# affordance_gpt4 = compute_affordance_VLM_GPT4(cropped_image=cropped_image, affordance_classes=affordance_classes)



a =2
