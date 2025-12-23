import numpy
import pandas

##words = pandas.read_sql("data.csv", header=None) Imports Data
words = ['Hello','Joey','how','are','you']

##Unigram Stemming

unigram_list = []

for word in words:
  for letter in word:
    unigram_list.append(letter)

'''Check dictionary
if unigram in dictionary:
  return "already in dictionary"
  break
else:
  add to dictionary
'''

print(unigram_list)

##Bigram Stemming

bigram_list = []

for word in words:
  modified_word = word[:-1]  # Exclude the last character
  for i in range(len(modified_word) - 1):
    bigram = modified_word[i:i+2]
    bigram_list.append(bigram)

'''Check dictionary
if bigram in dictionary:
  return "already in dictionary"
  break
else:
  add to dictionary
'''

print(bigram_list)
