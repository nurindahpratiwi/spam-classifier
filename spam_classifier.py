# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Download NLTK resources for stopwords and tokenization
nltk.download('stopwords')
nltk.download('punkt')

# Read CSV file into a Pandas DataFrame
data = pd.read_csv('/home/wiwaaw/Projects/powerfulProjects/spam-class/spam_emails.csv', encoding='ISO-8859-1', encoding_errors='strict')

# Initialize Porter Stemmer for stemming
ps = PorterStemmer()

# Define a function to transform text data
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    text = word_tokenize(text)

    y = []
    # Keep only alphanumeric tokens
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Define a function to preprocess text data
def preprocess(data):
    data = data.str.replace('escapenumb', '')
    data = data.str.replace('\n', ' ')
    data = data.str.replace('Â', '')
    data = data.str.replace('\d', '')
    data = data.str.replace('[^\w\s]', '')
    return data

# Define a function to predict using the trained model
def predict(email: str) -> float:
    vectorized_email = tfidf.transform([email]).toarray()
    prediction = mnb.predict(vectorized_email)[0]
    return prediction

# Initialize TF-IDF Vectorizer with a maximum of 3000 features
tfidf = TfidfVectorizer(max_features=3000)

# Transform the 'email' column into a TF-IDF matrix
X = tfidf.fit_transform(data['email']).toarray()
y = data['spam'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Multinomial Naive Bayes classifier
mnb = MultinomialNB()
# Train the classifier on the training data
mnb.fit(X_train, y_train)
# Predictions on the training and testing sets
mnbtrain = mnb.predict(X_train)
mnbtest = mnb.predict(X_test)

# Define a function to report accuracy
def acc_report(actual, predicted):
    acc_score = accuracy_score(actual, predicted)
    #confusion matrix and classification report
    cm_matrix = confusion_matrix(actual, predicted)
    class_rep = classification_report(actual, predicted)
    print('The accuracy of the model is ', acc_score)
    # Print confusion matrix and classification report if needed
    print(cm_matrix)
    print(class_rep)

# Report accuracy on the training set
print("Training Set:")
acc_report(y_train, mnbtrain)

# Report accuracy on the testing set
print("Testing Set:")
acc_report(y_test, mnbtest)

# Save the trained model and TF-IDF Vectorizer for future use
pickle.dump(mnb, open('spam.pkl', 'wb'))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
