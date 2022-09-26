from sklearn import svm
import spacy

class Category:
    BOOKS = "BOOKS"
    CLOTHING = "CLOTHING"
     
x_data = ["i love the book",
          "this is a great book",
          "the fit is great",
          "i love the shoes"]
y_data = [Category.BOOKS, Category.BOOKS, Category.CLOTHING, Category.CLOTHING]

nlp = spacy.load("en_core_web_lg") #model of the word vectors 
doc = [nlp(text) for text in x_data] #build the word vectors of each sentence

clf_svm = svm.SVC(kernel="linear") #build a classifier
clf_svm.fit([x.vector for x in doc], y_data)

test_data = [
    "i like the book",
    "the shoes is great",
    "i love the pants",
    "the book is great",
    "shoes are alright",
    "i love the books",
    "the story is great",
    "the outfit is great"
]

#transform each sentence to word vectors and predict the result 
result = [clf_svm.predict([nlp(sentence).vector]) for sentence in test_data]
for i in result:
     print(i)

#result
'''
['BOOKS']
['CLOTHING']
['CLOTHING']
['BOOKS']
['CLOTHING']
['BOOKS']
['BOOKS']
['CLOTHING']
'''