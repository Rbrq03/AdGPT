from openai import OpenAI
from zhipuai import ZhipuAI
import time


def LLMs_chat(messages, api_key, base_url, model, llm, temperature=0.6):

    if llm == "openai":
        client = OpenAI(api_key=api_key, base_url=base_url)
    elif llm == "glm":
        client = ZhipuAI(api_key=api_key)

    try:
        print(messages)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    except:
        time.sleep(2)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

    response = response.choices[0].message.content

    return response
