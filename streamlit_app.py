import pickle
import streamlit as st
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

def main():
    tfidf_path = '/home/wiwaaw/Projects/powerfulProjects/spam-class/vectorizer.pkl'
    model_path = '/home/wiwaaw/Projects/powerfulProjects/spam-class/spam.pkl'

    # Load CountVectorizer and model
    tfidf = pickle.load(open(tfidf_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))

    st.title("SPAM EMAIL CLASSIFIER")

    input_text = st.text_area("Input text email")

    ps = PorterStemmer()
    
    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)

        y = []
        for i in text:
            if i.isalnum():
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            y.append(ps.stem(i))

        return " ".join(y)

    def preprocess(data):
        data = data.replace('escapenumber', '')
        data = data.replace('\n', ' ')
        data = data.replace('Â', '')
        data = data.replace('\d', '')
        data = data.replace('[^\w\s]', '')
        return data

    def predict(email: str) -> float:
        # Check if the CountVectorizer is fitted; if not, fit it on training data
        if not hasattr(tfidf, 'vocabulary_'):
            st.warning("CountVectorizer not fitted; fitting on training data.")
            return None

        vectorized_email = tfidf.transform([email])
        prediction = model.predict(vectorized_email)[0]
        return prediction

    if st.button('Detect'):
        transformed_text = transform_text(input_text)
        preprocessed_data = preprocess(transformed_text)
        
        # Check if the CountVectorizer is fitted; if not, fit it on training data
        if not hasattr(tfidf, 'vocabulary_'):
            st.warning("CountVectorizer not fitted; fitting on training data.")
            tfidf.fit([preprocessed_data])

        prediction = predict(preprocessed_data)

        if prediction is not None:
            st.write("Predicted Spam Probability:", prediction)
            if prediction == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")

if __name__ == '__main__':
    main()
