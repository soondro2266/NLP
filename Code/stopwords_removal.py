from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

phrase = "Here is an example sentence demonstrating the removal of stopwords"
stop_words = stopwords.words('english')

words = word_tokenize(phrase)

stripped_phrase = []
for word in words:
    if word not in stop_words:
        stripped_phrase.append(word)

print(" ".join(stripped_phrase))

#result
"""Here example sentence demonstrating removal stopwords"""