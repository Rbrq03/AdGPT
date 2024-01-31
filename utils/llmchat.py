from openai import OpenAI
import time

def ChatGPT_chat(messages, api_key, base_url, model, temperature=0.6):

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    except:
        time.sleep(2)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

    response = response.choices[0].message.content
    
    return response