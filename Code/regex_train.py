import re

regexp = re.compile(r"read|story|book") #無句首句尾

phrases = ["I like that story", "I love the book", "this hat is nice"]

match_phrases = []
search_phrases = []
for phrase in phrases:
    if re.match(regexp,phrase):
        match_phrases.append(phrase)
    if re.search(regexp,phrase):
        search_phrases.append(phrase)

print(match_phrases)
print(search_phrases)

#result
"""
[]
['I like that story', 'I love the book']
"""