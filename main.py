import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming X_train, y_train, X_test, y_test are your preprocessed data
# X_train and X_test are feature matrices, y_train and y_test are labels
def get_tfidf_matrix(csv_path):
    # Load the data
    data = pd.read_csv(csv_path)

    # Extract comments from the data
    comments = data['comment_text']

    # Initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the comments and transform into TF-IDF features
    tfidf_matrix = tfidf_vectorizer.fit_transform(comments)

    return tfidf_matrix

# Initialize parameters
num_epochs = 100
learning_rate = 0.01
num_classes = len(np.unique(y_train))
num_features = X_train.shape[1]




# Convert labels to one-hot encoding
def one_hot_encode(labels, num_classes):
    encoded = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        encoded[i, label] = 1
    return encoded

y_train_encoded = one_hot_encode(y_train, num_classes)

# Initialize weights and biases
weights = np.random.rand(num_features, num_classes)
bias = np.zeros((1, num_classes))

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    scores = np.dot(X_train, weights) + bias
    probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # Calculate loss
    loss = -np.sum(y_train_encoded * np.log(probabilities)) / len(X_train)
    
    # Backpropagation
    grad = probabilities - y_train_encoded
    dW = np.dot(X_train.T, grad) / len(X_train)
    db = np.sum(grad, axis=0, keepdims=True) / len(X_train)
    
    # Update weights and biases
    weights -= learning_rate * dW
    bias -= learning_rate * db

# Prediction on test set
test_scores = np.dot(X_test, weights) + bias
predicted_probabilities = np.exp(test_scores) / np.sum(np.exp(test_scores), axis=1, keepdims=True)
predicted_labels = np.argmax(predicted_probabilities, axis=1)

# Evaluate accuracy
accuracy = np.mean(predicted_labels == y_test)
print(f"Accuracy: {accuracy}")


# Example corpus (list of text documents)
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer to the data and transform the corpus into TF-IDF features
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Get the feature names (words in the vocabulary)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense numpy array (if needed)
tfidf_matrix_dense = tfidf_matrix.toarray()

# Print the feature names and TF-IDF matrix
print("Feature names:", feature_names)
print("TF-IDF Matrix:")
print(tfidf_matrix_dense)
