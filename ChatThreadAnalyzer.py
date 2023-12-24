import json
import sys
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Spacy setup
nlp = spacy.load("en_core_web_sm")

# Transformers setup
classifier = pipeline("zero-shot-classification")

emoticon_dict = {
    "🙂": "positive",
    "😊": "positive",
    "😀": "positive",
    "😁": "positive",
    "😂": "positive",
    "🤣": "positive",
    "😍": "positive",
    "😘": "positive",
    "😚": "positive",
    "😋": "positive",
    "😜": "positive",
    "😛": "positive",
    "🤪": "positive",
    "😎": "positive",
    "🥳": "positive",
    "😇": "positive",
    "🤤": "positive",
    "😔": "negative",
    "😞": "negative",
    "😟": "negative",
    "😠": "negative",
    "😡": "negative",
    "🤬": "negative",
    "😢": "negative",
    "😭": "negative",
    "😤": "negative",
    "😩": "negative",
    "😫": "negative",
    "😒": "negative",
    "🙁": "negative",
    "😕": "negative",
    "😖": "negative",
    "😨": "negative",
    "😰": "negative",
    "😥": "negative",
    "🤯": "negative",
    "😐": "neutral",
    "😑": "neutral",
    "😶": "neutral",
    "🤔": "neutral",
    "🤐": "neutral",
    "😬": "neutral",
    "🙄": "neutral",
    "😯": "neutral",
    "😦": "neutral",
    "😧": "neutral",

    # Surprised or shocked
    "😲": "surprise",
    "😳": "surprise",
    "🤯": "surprise",

    # Confused or unsure
    "😕": "confused",
    "😖": "confused",
    "🤨": "confused",

    # Love and affection
    "😍": "love",
    "😘": "love",
    "😚": "love",
    "❤️": "love",
    "💕": "love",
    "💖": "love",
    "💗": "love",
    "💓": "love",
    "💞": "love",
    "💘": "love",
    "💝": "love",

    # Fear
    "😨": "fear",
    "😰": "fear",
    "😱": "fear",

    # Disgust
    "🤢": "disgust",
    "🤮": "disgust",

    # Skeptical or disapproving
    "🙄": "skeptical",
    "😒": "skeptical",

    # Tired or sick
    "😴": "tired",
    "🤒": "sick",
    "🤕": "sick",
    "🤢": "sick",
    "🤮": "sick",

    # Other emotions or states
    "😵": "dizzy",
    "🥴": "woozy",
    "🤠": "playful",
    "😷": "sick"
}


def parse_date(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d, %I:%M:%S\u202f%p')

def merge_consecutive_messages(messages):
    merged_messages = []
    current_message = messages[0]

    for next_message in messages[1:]:
        if next_message['author'] == current_message['author'] and is_close_in_time(current_message, next_message):
            current_message['message'] += ' ' + next_message['message']
        else:
            merged_messages.append(current_message)
            current_message = next_message

    merged_messages.append(current_message)
    return merged_messages

def is_close_in_time(msg1, msg2, max_gap=timedelta(minutes=10)):
    time1 = parse_date(msg1['date_time'])
    time2 = parse_date(msg2['date_time'])
    return abs(time1 - time2) <= max_gap

def thread_messages(messages):
    threads = []
    current_thread = [messages[0]]

    for next_message in messages[1:]:
        if is_continuation_of_thread(next_message, current_thread):
            current_thread.append(next_message)
        else:
            threads.append(current_thread)
            current_thread = [next_message]

    if current_thread:
        threads.append(current_thread)

    return threads

def is_continuation_of_thread(next_message, current_thread):
    last_message = current_thread[-1]
    return is_close_in_time(next_message, last_message)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def format_threads_for_json(threads):
    formatted_threads = []
    for thread in threads:
        formatted_thread = []
        for message in thread:
            formatted_message = {
                "date_time": message['date_time'],
                "author": message['author'],
                "message": message['message']
            }
            formatted_thread.append(formatted_message)
        formatted_threads.append(formatted_thread)
    return formatted_threads

def write_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def get_emoticon_sentiment(message):
    sentiment_count = {}

    # Count the occurrences of each sentiment based on emoticons in the message
    for emoticon, sentiment in emoticon_dict.items():
        if emoticon in message:
            if sentiment in sentiment_count:
                sentiment_count[sentiment] += 1
            else:
                sentiment_count[sentiment] = 1

    # Find the most common sentiment
    if sentiment_count:
        most_common_sentiment = max(sentiment_count, key=sentiment_count.get)
        return most_common_sentiment

    return None


def analyze_sentiment(message):
    emoticon_sentiment = get_emoticon_sentiment(message)

    if emoticon_sentiment:
        print(f"Used emoticon sentiment {emoticon_sentiment} ")
        return emoticon_sentiment

    print("Using sentiment intensity analyzer")

    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(message)
    compound_score = scores['compound']

    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def is_question(message):
    doc = nlp(message)
    for sent in doc.sents:
        if any(token.tag_ == "WP" or token.tag_ == "WRB" for token in sent):
            return True
    return False

def classify_message(message, candidate_labels):
    print(f"Classify message: `{message}`")
    result = classifier(message, candidate_labels)
    print(f"`{message}` classified as: {result['labels']}")
    return {label: score for label, score in zip(result['labels'], result['scores'])}

def format_threads_for_json(threads):
    candidate_labels = [
        "Informative", "Casual", "Sarcasm", "Happy", "Excited", "Amused", "Joyful", "Proud", 
        "Content", "Grateful", "Relieved", "Hopeful", "Inspired", "Amazed", "Surprised", 
        "Confused", "Curious", "Indifferent", "Bored", "Tired", "Annoyed", "Angry", "Frustrated", 
        "Disappointed", "Worried", "Anxious", "Scared", "Sad", "Heartbroken", "Grieving", 
        "Shocked", "Disgusted", "Distrustful", "Skeptical", "Jealous", "Embarrassed", 
        "Guilty", "Ashamed", "Nostalgic", "Sentimental", "Melancholic", "Lonely", "Overwhelmed",
        "Playful", "Teasing", "Mischievous", "Bantering", "Whimsical", "Ironic", "Witty", 
        "Sardonic", "Jocular", "Tongue-in-cheek", "Silly", "Mock-serious", "Sarcastic", 
        "Affectionate", "Complimentary", "Encouraging", "Empathetic", "Caring", "Sympathetic", 
        "Nostalgic", "Reflective", "Meditative", "Philosophical", "Mystified"
    ]

    print(f"Classifying with labels {candidate_labels}")

    formatted_threads = []

    # Function to process a single message
    def process_message(message):
        sentiment_tag = analyze_sentiment(message['message'])
        classification_results = classify_message(message['message'], candidate_labels)
        question_tag = "Question" if is_question(message['message']) else "Statement"
        return {
            "date_time": message['date_time'],
            "author": message['author'],
            "message": message['message'],
            "sentiment": sentiment_tag,
            "classification": classification_results,
            "question": question_tag
        }

    with ThreadPoolExecutor() as executor:
        for thread in threads:
            # Submit all messages in the thread to the executor
            future_to_message = {executor.submit(process_message, message): message for message in thread}
            formatted_thread = []
            for future in as_completed(future_to_message):
                # Collect results as they complete
                formatted_thread.append(future.result())
            formatted_threads.append(formatted_thread)

    return formatted_threads


def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <path_to_input_json_file.json> <path_to_output_json_file.json>")
        sys.exit(1)

    input_json_file_path = sys.argv[1]
    output_json_file_path = sys.argv[2]

    chat_data = load_data(input_json_file_path)
    merged_chat_data = merge_consecutive_messages(chat_data)
    threaded_conversations = thread_messages(merged_chat_data)
    formatted_threads = format_threads_for_json(threaded_conversations)
    write_to_json(formatted_threads, output_json_file_path)

    print(f"Threaded chat data has been written to '{output_json_file_path}'")

if __name__ == "__main__":
    main()