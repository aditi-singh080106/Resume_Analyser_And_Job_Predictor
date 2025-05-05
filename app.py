import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
from PIL import Image
import pytesseract

# Load pre-trained model and TF-IDF vectorizer (ensure these are saved earlier)
svc_model = pickle.load(open('clf.pkl', 'rb'))  # Example file name, adjust as needed
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Example file name, adjust as needed
le = pickle.load(open('encoder.pkl', 'rb'))  # Example file name, adjust as needed
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    # Try using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    return text


# Function to extract text from image using OCR
def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    elif file_extension in ['jpg', 'jpeg', 'png']:
        text = extract_text_from_image(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, TXT, or image file (jpg, jpeg, png).")
    return text


# Function to predict the category of a resume
def pred(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = cleanResume(input_resume)

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = svc_model.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]  # Return the category name


# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, DOCX, or image format and get the predicted job category.")

    # File upload section
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Extract text from the uploaded file
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")

            # Display extracted text (optional)
            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Make prediction
            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()




# import streamlit as st 
# import pickle 
# import re 
# import nltk
# import string
# import pandas as pd 
# from nltk.stem.porter import PorterStemmer 
# nltk.download('punkt')
# nltk.download('stopwords')

# # loading clf models 
# classifier = pickle.load(open('classifier.pkl','rb'))
# vectorizer = pickle.load(open('vectorizer.pkl','rb'))

# port_stem = PorterStemmer()
# # removing the unwanted text from data
# def stem_clean_resume(resume_text):
#   stem_txt = re.sub(r'[^a-zA-Z]',' ',resume_text)
#   stem_txt = re.sub(r'http\S+\s',' ',stem_txt)
#   stem_txt = re.sub(r'RT|cc',' ',stem_txt)
#   stem_txt = re.sub(r'#\S+\s',' ',stem_txt)
#   stem_txt = re.sub(r'@\S+',' ',stem_txt)
#   stem_txt = re.sub(r'[%s]'%re.escape(string.punctuation),' ',stem_txt)
#   stem_txt = re.sub(r'\s+',' ',stem_txt)
#   stem_txt = re.sub(r'[^\x00-\x7f]',' ',stem_txt)
#   stem_txt = stem_txt.lower()
#   stem_txt = stem_txt.split()
#   stem_txt = [port_stem.stem(word) for word in stem_txt if word not in stopwords.words('english')]
#   stem_txt = ' '.join(stem_txt)
#   return stem_txt


# # Creating Web application
# def main():
#     st.title("Resume Screening App")
#     upload_file = st.file_uploader("Upload Resume : ",type=['txt','pdf'])
#     if upload_file is not None:
#         try :
#             resume_bytes = upload_file.read()
#             resume_text = resume_bytes.decode('utf-8')
#         except UnicodeDecodeError:
#             # If UTF-8 decoding fails, try to decode using Latin-1
#             resume_text = resume_bytes.decode('latin-1')
#         clean_resume = stem_clean_resume(resume_text)
#          # trained model work on sparse matrix of resume dataset
#         sparsed_resume =vectorizer.transform([clean_resume])
#         prediction_id = classifier.predict(sparsed_resume)[0]
#         st.write(prediction_id)
#         # Map category Id to category name
#         category_mapping = {
#             15: 'Java Developer',
#             23: "Testing",
#             8 : 'DevOps Engineer',
#             20: 'Python Developer',
#             24: 'Web Designing',
#             12: 'HR',
#             13: 'Hadoop',
#             3 : 'Blockchain',
#             10: 'ETL Developer',
#             18: 'Operations Manager',
#             6 : 'Datascience' ,
#             22: 'Sales',
#             16: 'Mechanical Engineer',
#             1 : 'Arts' ,
#             7 : 'Database',
#             11: 'Electrical Engineer',
#             14: 'Health and Fitness',
#             19: 'PMO' ,
#             4 : 'Business Analyst' ,
#             9 : 'DotNet Developer' ,
#             2 : 'Automation Testing' ,
#             17: 'Network Security Engineer' ,
#             5 : 'Civil Engineer' ,
#             0 : 'Advocate' ,
#             21: 'SAP Developer'

#         }
#         category_name = category_mapping.get(prediction_id,"Unknown")
#         st.write("Predicted Category is : ",category_name)


# if __name__ == "__main__":
    # main()