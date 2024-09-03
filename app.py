from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for DOCX
import pytesseract  # Tesseract OCR for images
from PIL import Image
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

CORS(app)

# Text extraction functions
def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_image(file_path):
    """Extracts text from an image using OCR."""
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    return text

def extract_text(file_path):
    """Main function to extract text based on file type."""
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return extract_text_from_image(file_path)
    else:
        raise ValueError("Unsupported file format")

# Text preprocessing function
def preprocess_text(text):
    """Cleans and preprocesses the extracted text."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters and numbers
    tokens = word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize the text
    processed_text = ' '.join(tokens)
    return processed_text

# Similarity computation function
def compute_similarity(job_description, cv_texts):
    """Computes the similarity between the job description and CVs."""
    documents = [job_description] + cv_texts
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return cosine_similarities

@app.route('/score-resume', methods=['POST'])
def score_resume_api():
    if 'resume' not in request.files:
        return jsonify({'error': 'Resume file is required'}), 400

    resume_file = request.files['resume']
    job_description = request.form.get('job_description')

    if not job_description:
        return jsonify({'error': 'Job description is required'}), 400

    try:
        # Save the file to a temporary location
        file_path = os.path.join("tmp", resume_file.filename)
        resume_file.save(file_path)

        # Extract text from the resume
        resume_text = extract_text(file_path)

        # Preprocess both texts
        processed_resume_text = preprocess_text(resume_text)
        processed_job_description = preprocess_text(job_description)

        # Compute similarity
        similarity_scores = compute_similarity(processed_job_description, [processed_resume_text])
        score = similarity_scores[0]  # Get the similarity score for the first CV

        # Remove the temporary file after processing
        os.remove(file_path)

        return jsonify({'score': score})
    except Exception as e:
        # Print the full stack trace for debugging purposes
        import traceback
        traceback.print_exc()

        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False)
