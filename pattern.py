import numpy
import pandas
from collections import Counter

def generate_ngrams(word, n):
  '''
  Args:
    word(str): The input word.
    n (int): The size of the n-gram.

  Returns:
    list: A list of character n-grams.
  '''
  ngrams = []
  if n <= 0 or n > len(word):
    return ngrams

  for i in range(len(word) - n + 1):
    ngram = word[i : i + n]
    ngrams.append(ngram)
  return ngrams

##words = pandas.read_sql("data.csv", header=None) Imports Data
words = ['Hello','Joey','how','are','you','doing','today']


#Iterate through the N-grams for printing
for n_value in range(1,7):
  print(f"\n{n_value}-grams (n={n_value}):")
  current_n_grams_list = []
  for word in words:
    ngrams_for_word = generate_ngrams(word, n_value)
    current_n_grams_list.extend(ngrams_for_word)
  print(current_n_grams_list)


#N-gram Frequency

all_ngram_frequencies = {}

# Iterate through n-gram lengths
for n_value in range(1, 5):
  current_n_grams = []
  # Iterate through each word in the words list
  for word in words:
    ngrams_for_word = generate_ngrams(word, n_value)
    current_n_grams.extend(ngrams_for_word)

  # Calculate frequencies
  all_ngram_frequencies[n_value] = Counter(current_n_grams)

#Results
for n, frequencies in all_ngram_frequencies.items():
  print(f"\nN-gram (n={n}) Frequencies:")
  print(frequencies)

