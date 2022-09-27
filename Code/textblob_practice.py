from textblob import TextBlob

phrase = "thje bookk was horrriblee"

tb_phrase = TextBlob(phrase)

print(tb_phrase.correct())