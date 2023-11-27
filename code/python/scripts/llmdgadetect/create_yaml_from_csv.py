import csv
import yaml
import argparse
import random

# Set up argument parsing
parser = argparse.ArgumentParser(description='Convert CSV file with domains to a YAML file')
parser.add_argument('-i', '--input', required=True, help='Input CSV file path')
parser.add_argument('-o', '--output', required=True, help='Output YAML file path')
parser.add_argument('--sample_size', type=int, default=None, help='Number of rows to sample from the CSV file')

# Parse arguments
args = parser.parse_args()
csv_file_path = args.input
yaml_file_path = args.output
sample_size = args.sample_size

# Read the CSV file and store the data
domains = []
with open(csv_file_path, mode='r') as file:
    csv_reader = list(csv.DictReader(file))
    # Sample rows if sample_size is specified and less than the total number of rows
    if sample_size and sample_size < len(csv_reader):
        csv_reader = random.sample(csv_reader, sample_size)

    for row in csv_reader:
        domain = row['domain']
        result = row['result']
        # Use actual newline characters in the string
        domain_entry = f"{domain}\n{{ 'domain': '{domain}', 'result': '{result}' }}\n"
        domains.append(domain_entry)

# Join the domains with actual newlines
#yaml_content = "".join(domains).strip()
yaml_content = "|" + "\n".join(domains).rstrip()
#yaml_content = "|\n" + "".join(domains).strip()

# Prepare the data for the YAML file
yaml_data = {
    'examples': [{'prompt': yaml_content}]
}

# Write the data to a YAML file
with open(yaml_file_path, 'w') as file:
    yaml.dump(yaml_data, file, default_flow_style=False, allow_unicode=True)

