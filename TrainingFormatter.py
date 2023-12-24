import json
import sys

def format_data_for_training(input_file, output_dir):
    with open(input_file, 'r', encoding='utf-8') as file:
        threads = json.load(file)

    # Iterate over each thread
    for thread in threads:
        # Dictionary to hold concatenated messages for each participant
        context_for_participant = {}

        # Iterate over each message in the thread
        for message in thread:
            author = message['author']
            text = message['message']

            # Update the context for each participant except the author of the current message
            for participant in context_for_participant:
                if participant != author:
                    context_for_participant[participant] += '\n' + text

            # If the author is not in the dictionary, add them
            if author not in context_for_participant:
                context_for_participant[author] = text

            # Write input-output pairs to the file for training
            for participant, context in context_for_participant.items():
                if participant != author:
                    with open(f"{output_dir}/{participant}_training_data.txt", "a", encoding="utf-8") as outfile:
                        outfile.write(f"Input: {context}\nOutput: {text}\n\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python TrainingFormatter.py <path_to_input_json_file> <output_directory>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    format_data_for_training(input_file, output_dir)

if __name__ == "__main__":
    main()
