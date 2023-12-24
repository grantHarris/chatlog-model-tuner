import re
import sys
import json

def parse_chat(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    messages = []
    for line in lines:
        match = re.match(r'\[(.*?)\] (.*?): (.*)', line)
        if match:
            date_time, author, message = match.groups()
            messages.append({'date_time': date_time, 'author': author, 'message': message})
        elif messages:
            # Append to the last message if the current line is a continuation
            messages[-1]['message'] += ' ' + line.strip()

    return messages

def save_to_file(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def main():
    if len(sys.argv) != 3:
        print("Usage: python normalize.py <input_chat_file.txt> <output_processed_file.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    parsed_data = parse_chat(input_file)
    save_to_file(parsed_data, output_file)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()
