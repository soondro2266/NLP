from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

class Category:
    BOOKS = "BOOKS"
    CLOTHING = "CLOTHING"
    
x = ["i love the book",
     "this is a great book",
     "the fit is great",
     "i love the shoes"]
y = [Category.BOOKS, Category.BOOKS, Category.CLOTHING, Category.CLOTHING]

vectorizer = CountVectorizer(binary=True) #build the vectorizer
x_vectors = vectorizer.fit_transform(x) #use x to make bag

clf_svm = svm.SVC(kernel='linear') #build a classifier
clf_svm.fit(x_vectors, y) 

test_data = [
    "i like the book",
    "the shoes is great",
    "i love the pants",
    "the book is great",
    "shoes are alright"
]

for text in test_data:
    vector = vectorizer.transform([text]) #transform test_data to vector
    result = clf_svm.predict(vector) #use vector to predict result
    print(result)

#result
'''
['BOOKS']
['CLOTHING']
['CLOTHING']
['BOOKS']
['CLOTHING']
'''