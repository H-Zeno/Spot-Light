import matplotlib.pyplot as plt
import numpy as np
import cv2
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import requests
import torch
import base64
from io import BytesIO
from PIL import Image
from typing import Dict

def compute_affordance_VLM_GPT4(cropped_image: np.ndarray, affordance_classes, api_key) -> int:
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

    response = GPT4_query(api_key, prmpt, cropped_image, max_tokens=1, detail="low")

    if response.status_code == 200:
        affordance_key = int(response.json()["choices"][0]["message"]["content"])
        return affordance_key
    else:
        print("Error:", response.status_code)

def GPT4_query(api_key, prompt, image, max_tokens: int = 1, detail: str="low"):
    """
    send query to GPT4 model via api

    @param api_key:
    @param prompt:
    @param image:
    @return:
    """
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
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": detail
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response

def compute_advanced_affordance_VLM_GPT4(cropped_image: np.ndarray, affordance_dict: Dict, api_key: str) -> Dict:
    """
    Takes in the cropped image of a light switch (bounding box) and predicts advanced affordance using VLM GPT4, returning a dict
    @param cropped_image:
    @param affordance_classes:
    @param api_key:
    @return:
    """

    # build prompt
    prompt = "Describe light switch: "
    for idx, (key, value) in enumerate(affordance_dict.items()):
        prompt += f"{key} ({'/'.join(value)})"
        if idx < len(affordance_dict) - 1:
            prompt += ", "
        else:
            prompt += ". "
    prompt += f"Format: <{'>, <'.join(list(affordance_dict.keys()))}>. answer all lower case, use no extra characters"

    response = GPT4_query(api_key, prompt, cropped_image, max_tokens=50, detail="low")

    if response.status_code == 200:
        values = response.json()["choices"][0]["message"]["content"].split(", ")
        keys = list(affordance_dict.keys())
        affordance_dict = dict(zip(keys, values))
        return affordance_dict
    else:
        print("Error:", response.status_code)


def GPT4_query(api_key, prompt, cropped_image, max_tokens: int = 1, detail: str="low"):
    """
    send query to GPT4 model via api

    @param api_key:
    @param prompt:
    @param image:
    @return:
    """
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
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": detail
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response

def compute_affordance_VLM_GPT4_test(cropped_image: np.ndarray, affordance_classes: Dict, api_key: str):

    # prompt = "Tell me what you see, how to interact with it, orientation, geometry. USE KEYWORDS No Sentences"
    # prompt = "Describe Light switch type (single/double, push, rotating, rocker), interaction, geometry/alignment. USE KEYWORDS NO SETENCES"
    # prompt = ("Describe light switch: button count (single/double/multi), switch type: (push button/rocker switch/rotating switch), button stacking (vetical/horizontal/None if single). Format: <button count>, <switch type>, <button stacking>")

    affordance_dict = {"switch type": ["push button switch", "rocker switch", "rotating switch"],
                       "button count": ["single", "double", "multi"],
                       "button stacking": ["vertical", "horizontal", "none"]}

    # build prompt
    prompt = "Describe light switch: "
    for idx, (key, value) in enumerate(affordance_dict.items()):
        prompt += f"{key} ({'/'.join(value)})"
        if idx < len(affordance_dict) - 1:
            prompt += ", "
        else:
            prompt += ". "
    prompt += f"Format: <{'>, <'.join(list(affordance_dict.keys()))}>"

    response = GPT4_query(api_key, prompt, cropped_image, max_tokens=50, detail="low")

    a = 2

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
def compute_affordance_VLM_test(model: str, affordance_classes: Dict[int, str], cropped_image: np.ndarray):

    if model=="GPT4":
        api_key = "..."
        affordance_key = compute_affordance_VLM_GPT4(cropped_image=cropped_image, affordance_classes=affordance_classes, api_key=api_key)
        return affordance_key
    if model == "LLAVA":
        affordance_key = compute_affordance_VLM_llava(cropped_image=cropped_image, affordance_classes=affordance_classes)
        return affordance_key
    else:
        print("Model not supported")



if __name__ == "__main__":

    affordance_dict = {"button type": ["push button switch", "rotating switch", "toggle switch", "none"],
                       "button count": ["single", "double", "none"],
                       "button position (wrt. other button!)": ["buttons stacked vertically", "buttons side-by-side", "none"],
                       "interaction inference from symbols": ["top/bot push", "left/right push", "center push", "no symbols present"]}



    cropped_image = cv2.imread("/home/cvg-robotics/tim_ws/GPT4_visuals/rocker_switch_2.png")

    plt.imshow(cropped_image)
    plt.show()

    api_key = "..."

    # affordance = compute_advanced_affordance_VLM_GPT4(cropped_image=cropped_image, affordance_dict=affordance_dict, api_key=api_key)
    # print(affordance)

    prompt = "given that you see a light switch, tell me at which pixel locations of the iamge (multiple is possible) the button can be interacted with, and what interaction i have to perform to turn it on. Use the format <x>, <y>, <interaction>"

    ans = GPT4_query(api_key, prompt, cropped_image, max_tokens=50, detail="low")

    a = 2