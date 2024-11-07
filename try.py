import json

def load_json_file(filepath):
    """Load JSON data from a file."""
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {filepath} is not a valid JSON.")
        return None

# Now load the JSON data
json_result = load_json_file('./faq.json')
print(json_result)
if json_result is None:
    print("Failed to load the JSON file.")

