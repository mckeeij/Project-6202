import sqlite3
import csv
import re
import os


DB_NAME = 'dict.db'

# Define the corpus file
CORPUS_FILE = 'C:\\Users\\Joeyj\\Proj6202\\output_words.txt'

def setup_database(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS words (
            word TEXT PRIMARY KEY
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database '{db_name}' and table 'words' set up to store unique words.")

def process_corpus(corpus_file, db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    with open(corpus_file, 'r') as f:
        text = f.read()

    # Seperates each word from .txt (alphabetic characters)
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    for word in words:
        # Inserts word into .db
        cursor.execute("INSERT OR IGNORE INTO words (word) VALUES (?)", (word,))

    conn.commit()
    conn.close()

def display_words(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT word FROM words ORDER BY word ASC")
    rows = cursor.fetchall()
    conn.close()

    if rows:
        print("\nUnique words in the database:")
        for row in rows:
            print(f"'{row[0]}'")
    else:
        print("\nNo unique words found in the database.")

# --- Execute the functions ---
# Setup the database
setup_database(DB_NAME)

# Process the corpus
process_corpus(CORPUS_FILE, DB_NAME)

# Display the unique words
display_words(DB_NAME)

# Data Purging steps
#os.remove(CORPUS_FILE)
#os.remove(DB_NAME)
