# LLMDGAgenerator

## Introduction
This script is a PoC for implementing a Domain Generation Algorithm (DGA) using the a LLM. It's designed to generate domain names based on provided keywords and certain constraints. The script uses the FastAPI framework for creating a simple web server to handle requests.

## Features
- Generates domain names based on two synonyms of a provided keyword.
- Removes a random character from the concatenated synonyms.
- Adds a Top-Level Domain (TLD) to the generated domain name.
- Outputs the result in JSON format.
- Includes input validation for the keyword and seed string.

## Requirements
- Python 3.x
- `fastapi` library
- `openai` library version 0.28
- `uvicorn` for running the server

## Installation

To install the required libraries, run:

```
pip install fastapi openai==0.28 uvicorn
```

Ensure you have the correct version of the `openai` library.

## Usage
Current implementation of the script uses a local LLM using [Zephyr Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)

So you just need to  point to an openAI compatible API :

```python
openai.api_key = "EMPTY"
openai.api_base = "https://chatbotapi.ingenieria.uncuyo.edu.ar"

```

To start the server, run the script:

```
python llmdgagen.py
```

This will start the server on `http://0.0.0.0:8000`.

### API Endpoint

The script exposes a single GET endpoint at `/dgagenerator`.

#### Parameters
- `seed_str`: A string to seed the generation process. Limited to 20 numbers.
- `number_str`: The number of DGA examples to generate.
- `keyword_str`: The keyword based on which domain names are generated. Limited to 12 characters.

#### Example Request

```
GET /dgagenerator?seed_str=1234567890&number_str=2&keyword_str=example
```

### Response Format

The response is in JSON format, with each domain in its own object:

```json
[
  {"domain": "exampledomain1.com"},
  {"domain": "exampledomain2.net"}
]
```

## Error Handling

The script provides basic error handling for input validation. If the `seed_str` contains more than 20 numbers, it defaults to a predefined string. Similarly, if the `keyword_str` is longer than 12 characters, it defaults to 'injection'.

## Contributing

Contributions to improve the script or add new features are welcome. Please submit a pull request or open an issue for discussion.

## License

MIT
