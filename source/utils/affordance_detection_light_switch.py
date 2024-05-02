import numpy as np
import cv2
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import requests
import torch
import base64
from io import BytesIO
from PIL import Image
from typing import Dict

def compute_affordance_VLM_GPT4(cropped_image: np.ndarray, affordance_classes, api_key):
    """
    Takes in the cropped image of a light switch (bounding box) and predicts the affordance using VLM GPT4

    @param cropped_image: np.ndarray
    @param affordance_classes: dict
    @param api_key: openAI api key
    @return: affordance_key: int, specifying the affordance class according to the affordance_classes dict
    """

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


def compute_affordance_VLM_llava(cropped_image, affordance_classes):

    processor_llava = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model_llava = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                                    torch_dtype=torch.float16,
                                                                    low_cpu_mem_usage=True)
    model_llava.to("cuda:0")

    prmpt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nIs this "

    for key, value in affordance_classes.items():
        if key < len(affordance_classes) - 1:
            prmpt += f"a {value} button light switch (if yes answer {key}) OR "
        else:
            prmpt += f"{value} which is unlikely (if yes answer {key})."
    prmpt += "<|im_end|><|im_start|>assistant\n"

    inputs = processor_llava(prmpt, cropped_image, return_tensors="pt").to("cuda:0")
    output = model_llava.generate(**inputs, max_new_tokens=100)

    affordance_key = int(processor_llava.decode(output[0], skip_special_tokens=True)[-2])

    del model_llava
    torch.cuda.empty_cache()

    return affordance_key
def test_compute_affordance_VLM(model: str, affordance_classes: Dict[int, str], cropped_image: np.ndarray):

    if model=="GPT4":
        api_key = "..."
        affordance_key = compute_affordance_VLM_GPT4(cropped_image=cropped_image, affordance_classes=affordance_classes, api_key=api_key)
    if model == "LLAVA":
        affordance_key = compute_affordance_VLM_llava(cropped_image=cropped_image, affordance_classes=affordance_classes)
    else:
        print("Model not supported")

    return affordance_key

if __name__ == "__main__":
    affordance_classes = {0: "SINGLE PUSH",
                          1: "DOUBLE PUSH",
                          2: "ROTATING",
                          3: "something else"}

    cropped_image = cv2.imread("/home/cvg-robotics/tim_ws/turn_button_spot.png")

    test_compute_affordance_VLM(model="LLAVA", affordance_classes=affordance_classes, cropped_image=cropped_image)