import streamlit as st
import pandas as pd
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Function to clean resume text
def clean_resume(txt):
    clean_text = re.sub('http\S+\s', ' ', txt)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+\s', ' ', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Function to extract text from a PDF file
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Loading dataset and preprocess the data
#@st.cache_data(allow_output_mutation=True)
def load_data():
    dataset = pd.read_csv("UpdatedResumeDataSet.csv")
    dataset['cleaned_resume'] = dataset['Resume'].apply(clean_resume)
    return dataset

# Training the model dynamically
#@st.cache_resource(allow_output_mutation=True)
def train_model(data):
    le = LabelEncoder()
    data['Category'] = le.fit_transform(data['Category'])

    X = data['cleaned_resume']
    y = data['Category']

    # Vectorizing text data
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Training a lightweight model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, vectorizer, le

# Loading and training the model
data = load_data()
model, vectorizer, label_encoder = train_model(data)

# Function to predict the category
def predict_category(resume_text):
    cleaned_text = clean_resume(resume_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    predicted_category = label_encoder.inverse_transform(prediction)
    return predicted_category[0]

# Streamlit app
def main():
    st.set_page_config(page_title="PDF Resume Category Predictor", page_icon="📄", layout="wide")
    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF format to get its predicted job category.")

    # File upload section
    uploaded_file = st.file_uploader("Upload a Resume (PDF format only)", type=["pdf"])

    if uploaded_file is not None:
        try:
            # Extract text from the uploaded PDF
            resume_text = extract_text_from_pdf(uploaded_file)
            st.write("Successfully extracted text from the uploaded resume.")

            # Display extracted text (optional)
            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Make prediction
            st.subheader("Predicted Category")
            category = predict_category(resume_text)
            st.success(f"The predicted category of the uploaded resume is: **{category}**")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
