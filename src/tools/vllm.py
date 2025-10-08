from openai import OpenAI
import os

import config
def vllm(prompt: str, image_path: str = "data/tools/test_image.png") -> str:
    """
    Visual Large Language Model (VLM) that can answer the user's questions based on the given image.
    Args:
        prompt (str): The prompt containing the user's question.
        image_path (str): The path to the image file.
    Returns:
        str: The response from the VLLM.
    """
    if config.is_sandbox():
        return "The vegetables appear to be fresh, as they are in their original packaging and have a vibrant color, which typically indicates they are not overripe or old."

    if not image_path:
        image_path = "data/tools/test_image.png"

    PORT = 8007
    client_vlm = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", PORT)),
    )
    messages = [{"role": "system", "content": prompt}]
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_path}},
        ],
    })
    result = client_vlm.chat.completions.create(messages=messages, model="test")
    result = result.choices[0].message.content

    return result

FUNCTIONS = {
    "vllm": vllm,
}

if __name__ == "__main__":
    prompt = "Detect the freshness of the vegetable."
    image_path = "data/tools/test_image.png"
    res = vllm(prompt, image_path)
    print(res)