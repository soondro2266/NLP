import re

regexp = re.compile(r"read|\bstory\b|\bbook\b")

phrases = ["i like the story",
           "i love the history",
           "the book is good",
           "he tread the bug"]

search_phrases = []
for phrase in phrases:
    if re.search(regexp,phrase):
        search_phrases.append(phrase)

print(search_phrases)

#result
"""
['i like the story', 'the book is good', 'he tread the bug']
"""