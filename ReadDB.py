import sqlite3
import os

DB_NAME = '3wn1.db'

def export_db_to_txt(db_name, output_txt_file):
  #Input your file path (beware of escape characters)
    conn = sqlite3.connect("C:\\Users\\Joeyj\\Downloads\\3wn1.db")
    cursor = conn.cursor()

    # Fetch all words from the 'words' table
    cursor.execute("SELECT word FROM words ORDER BY word ASC")
    rows = cursor.fetchall()
    conn.close()

    if rows:
        with open(output_txt_file, 'w') as f:
            for row in rows:
                f.write(row[0] + '\n') # Write each word followed by a newline
        print(f"Successfully exported words from '{db_name}' to '{output_txt_file}'")
    else:
        print(f"No words found in '{db_name}' to export.")

# Define the output text file name
OUTPUT_TXT_FILE = 'output_words.txt'

# Execute the function to convert the .db to .txt
export_db_to_txt(DB_NAME, OUTPUT_TXT_FILE)
