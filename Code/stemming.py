from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

phrase = "i was handsome"
words = word_tokenize(phrase)

print(words)

stemmed_words = []
for word in words:
    stemmed_words.append(stemmer.stem(word))

print(stemmed_words)

#result
"""
['reading', 'the', 'books']
['read', 'the', 'book']
"""