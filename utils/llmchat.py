from openai import OpenAI
import time

def ChatGPT_chat(messages, api_key, model="gpt-4-turbo-preview", temperature=0.6):

    client = OpenAI(api_key=api_key)
    
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