## DGA generator
## requires openai library == 0.28
from fastapi import FastAPI, Response
import openai
import argparse
import json
import re


parser = argparse.ArgumentParser(description="Run a FastAPI application for DGA Generation.")
parser.add_argument("--port", type=int, default=8000, help="Port of the service")

openai.api_key = "EMPTY"
openai.api_base = "https://chatbotapi.ingenieria.uncuyo.edu.ar/v1"

model = ""
system_prompt = """
You are a Domain Generation Algorithm (DGA). I will provided you with a 'keyword', and you will answer with a generated  domain name based on two synonyms of the provided 'keyword'.
The procedure for generating a new domain name is the following:
1. Pick two synonyms of the 'keyword' provided.
2. remove one random character
3. Concatenate the words.
4. Do not add a number to the resulting string.
5. Add a TLD domain

Here there are two examples:
```
Example 1:
'keyword': pets
picked synonym 1: mittens
picked synonym 2: pooches
domain generated: poochesmitens.bz

final answer after removing one character:
{"domain": "poochesmitens.bz"},

Example 2:
'keyword': wealth
picked synonym 1: riches
picked synonym 2: fortune
domain generated: richefortune.com

final answer after removing one character:
{"domain": "richefortune.com"},
```

You answer should be JSON in the following format:
```
[
{"domain": "richefortune.com"},
{"domain": "poochesmitens.bz"}
]
```

NEVER answer anything but JSON. Do not provide any explanation.
"""



def contains_up_to_20_numbers(s):
    numbers = re.findall(r'\d', s)
    return len(numbers) <= 20


app = FastAPI()

@app.get("/dgagenerator")
async def dga_generator(seed_str: str, number_str:str, keyword_str:str):
    """
    
    """
    dgalist="{}"
    try:
        # Parse the input date string
        if contains_up_to_20_numbers(seed_str) != True:
            seed_str = "0"*20
        if len(keyword_str) > 12:
            keyword_str="injection"    
        dga_generator_prompt = f"""
        Given the seed:{seed_str}, and the key word: {keyword_str} provide me with {number_str} DGA examples. Do not explain anything.

        """
        # create a chat completion
        completion = openai.ChatCompletion.create(
        model = "",
        temperature = 0,
        top_k = 2,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user","content": dga_generator_prompt}
                ]
        )
        response_str = completion.choices[0].message.content
        print(response_str)
        if response_str is not None:
            try:
                dgalist = json.loads(response_str)

            except ValueError:
                try:
                    response_str = response_str[response_str.find("["):response_str.find("]")+1]
                    print(response_str)
                    dgalist = json.loads(response_str)
                except ValueError as e:
                    print("Something went wrong here: load")
                    print("Error:", e)
        return dgalist
    except ValueError:
        return Response("Invalid date format", status_code=400)

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)