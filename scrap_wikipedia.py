# 1_scraper.py
import wikipedia
import re
import csv
import random

wikipedia.set_lang("en")

topics = {
    "Math": "Mathematics",
    "Science": "science",  
    "History": "History",
    "English": "English language"
}

max_sentences_per_category = 30
output_file = "educational_dataset.csv"

category_data = {}

for label, page_title in topics.items():
    print(f"Scraping page: {page_title} for label: {label}")
    sentences = []

    try:
        page = wikipedia.page(page_title, auto_suggest=False)
        content = page.content
    except wikipedia.DisambiguationError as e:
        print(f"Disambiguation error for {page_title}: {e.options}")
        continue
    except Exception as e:
        print(f"Failed to fetch {page_title}: {e}")
        continue

    raw_sentences = re.split(r'(?<=[.!?]) +', content)

    for sent in raw_sentences:
        sent = sent.strip()
        if 30 < len(sent) < 150 and sent[0].isupper() and not sent.startswith('=='):
            sentences.append((sent, label))
        if len(sentences) >= max_sentences_per_category:
            break

    category_data[label] = sentences
    print(f"Collected {len(sentences)} sentences for {label}")

# Combine and shuffle
all_data = []
for label in topics:
    all_data.extend(category_data.get(label, []))

random.shuffle(all_data)

# Save to CSV
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["text", "label"])
    writer.writerows(all_data)

print(f"\nDataset saved as '{output_file}'")  

