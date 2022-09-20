from sklearn.feature_extraction.text import CountVectorizer

train_data = [
    "i love the book",
    "this is a great book",
    "the fit is great",
    "i love the shoes"
]

vectorizer = CountVectorizer(binary=True)
vectors = vectorizer.fit_transform(train_data)
print(vectorizer.get_feature_names())
print(vectors.toarray())