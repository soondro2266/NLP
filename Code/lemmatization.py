import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].lower()
    if tag not in ['n', 'v', 'a', 'r']:
        return "n"
    return tag

lemmatizer = WordNetLemmatizer()
phrase = "The striped bats are hang on their feet for best"
words = word_tokenize(phrase)

lemmatized_words = []
for word in words:
    lemmatized_words.append(lemmatizer.lemmatize(word, get_wordnet_pos(word)))

print(lemmatized_words)

#result
"""
['The', 'strip', 'bat', 'be', 'hang', 'on', 'their', 'foot', 'for', 'best']
"""