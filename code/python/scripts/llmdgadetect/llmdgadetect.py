from fastapi import FastAPI, Response
import argparse
import openai
import json
import yaml
import uvicorn

parser = argparse.ArgumentParser(description="Run a FastAPI application for DGA detection.")
parser.add_argument("--yaml_file", type=str, required=True, help="Path to the YAML file containing domain examples.")
parser.add_argument("--port", type=int, default=8000, help="Port of the service")

args = parser.parse_args()

app = FastAPI()

model = ""
system_prompt = """
you are a Domain Generation Algorithm detector. 
You will recieve a domain names and you should answer if the domains is `dga` or `benign`. Your answer
should be in JSON. Please answer nothing but the JSON. Do not explain anything.



"""

openai.api_key = "EMPTY"
openai.api_base = "https://chatbotapi.ingenieria.uncuyo.edu.ar/v1"


# Read the YAML file
def read_domains(yaml_file):
    
    with open(yaml_file, 'r') as file:
        domains_data = yaml.safe_load(file)
    examples = domains_data['examples'][0]
    examples = examples['prompt']
    return examples

@app.get("/dgadetector")
async def dga_generator(domain_str:str):
    """
    
    """
    dga_answer={}
    try:
        if len(domain_str) > 40:
            domain_str="injection"    
        dga_detector_prompt = domain_str 
        
        # create a chat completion
        completion = openai.ChatCompletion.create(
        model = "",
        temperature = 0,
        top_k = 2,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user","content": dga_detector_prompt}
                ]
        )
        response_str = completion.choices[0].message.content
        print(response_str)
        if response_str is not None:
            try:
                dga_answer = json.loads(response_str)

            except ValueError:
                try:
                    response_str = response_str[response_str.find("["):response_str.find("]")+1]
                    print(response_str)
                    dga_answer = json.loads(response_str)
                except ValueError as e:
                    print("Something went wrong here: load")
                    print("Error:", e)
        return dga_answer
    except ValueError:
        return Response("Invalid date format", status_code=400)

# Run the app
if __name__ == "__main__":
    examples = read_domains(args.yaml_file)
    system_prompt += examples
    uvicorn.run(app, host="0.0.0.0", port=args.port)
