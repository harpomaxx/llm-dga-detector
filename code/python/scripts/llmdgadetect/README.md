# LLM DGA Detector

## Overview
This script is a PoC for implementing a Domain Generation Algorithm (DGA) detector using the a LLM. DGAs are often used by malware to generate a large number of domain names as rendezvous points with their command and control servers, making their detection crucial for cybersecurity.

## Features
- FastAPI for an efficient and easy-to-use web framework.
- Integration with OpenAI's language models for accurate DGA detection.
- YAML configuration for easy setup and customization.

## Prerequisites
- Python 3.6+
- FastAPI
- Uvicorn
- PyYAML
- OpenAI API key

## Installation
1. Clone this repository.
   ```
   git clone llm-dga-detector
   cd llm-dga-detector
   ```
2. Install the required packages.
   ```
   pip install fastapi uvicorn pyyaml openai
   ```

## Configuration
Create a YAML file following the structure provided in `llm_dga_detector_prompt.yml`. This file should contain examples of domain names for the model to learn from.

## Usage
Run the FastAPI application with the following command:
```
python app.py --yaml_file llm_dga_detector_prompt.yml  --port 8000
```
The service will be available at `http://localhost:8000/dgadetector`.

## API Endpoints
- `GET /dgadetector`: Takes a `domain_str` as a query parameter and returns whether the domain is `dga` or `benign`.

## Examples
To test the service, you can use a tool like `curl`:
```
curl "http://localhost:8000/dgadetector?domain_str=example.com"
```

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the [MIT License](LICENSE).

