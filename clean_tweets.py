import json
import re
from pathlib import Path

def clean_tweet_text(text: str) -> str:
    """Strips URLs, user mentions, and trailing whitespace from tweet text."""
    # Remove URLs (http, https, t.co links)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove Twitter handles (@username)
    text = re.sub(r'@\w+', '', text)
    # Collapse multiple spaces/newlines into a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_twitter_archive(input_path: str, output_json: str, output_jsonl: str):
    file_path = Path(input_path)
    
    if not file_path.exists():
        print(f"Error: Could not find '{input_path}'. Please ensure your file is in the correct directory.")
        return

    print(f"Reading {input_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Twitter archive js files start with 'window.YTD.tweets.part0 = ['
        if '=' in content:
            content = content.split('=', 1)[1]

    raw_tweets = json.loads(content)
    cleaned_tweets = []
    formatted_dataset = []

    for item in raw_tweets:
        tweet = item.get('tweet', {})
        full_text = tweet.get('full_text', '')

        # Filter out retweets, replies, and empty/short posts
        is_retweet = full_text.startswith('RT @')
        is_reply = tweet.get('in_reply_to_status_id') is not None
        
        if not is_retweet and not is_reply:
            cleaned = clean_tweet_text(full_text)
            
            # Keep posts with substantive text (e.g., > 30 characters)
            if len(cleaned) >= 30:
                cleaned_tweets.append(cleaned)
                formatted_dataset.append({
                    "instruction": "Write a short post reflecting my personal voice, style, and technical perspective.",
                    "input": "",
                    "output": cleaned
                })

    # 1. Save raw array of strings
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(cleaned_tweets, f, indent=2, ensure_ascii=False)

    # 2. Save JSONL format ready for Unsloth / Hugging Face SFT fine-tuning
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in formatted_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("\nProcessing complete!")
    print(f"• Total valid original posts extracted: {len(cleaned_tweets)}")
    print(f"• Saved raw cleaned list to: {output_json}")
    print(f"• Saved fine-tuning dataset to: {output_jsonl}")

if __name__ == "__main__":
    # Update 'data/tweets.js' if your file path is located elsewhere
    process_twitter_archive(
        input_path='data/tweets.js',
        output_json='cleaned_tweets.json',
        output_jsonl='train_tweets.jsonl'
    )