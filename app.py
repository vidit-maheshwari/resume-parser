from flask import Flask, request, jsonify
import google.generativeai as genai
from PyPDF2 import PdfReader
import docx2txt
import os
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ResumeParser:
    def __init__(self, api_key):
        # Configure Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF files"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX files"""
        try:
            text = docx2txt.process(file_path)
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")

    def extract_text(self, file_path):
        """Extract text based on file extension"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        else:
            raise Exception(f"Unsupported file format: {file_extension}")

    def parse_resume(self, text):
        """Parse resume text using Gemini API"""
        prompt = """
        Extract information from the following resume and format it as JSON with the following structure:
        {
          "profile": {
            "location": {
              "current": "",
              "relocation": ""
            },
            "certifications": {
              "select": "",
              "uploadOrUrl": ""
            },
            "education": {
              "college": "",
              "degree": "",
              "stream": ""
            },
            "professionalExperience": [
              {
                "company": "",
                "from": "",
                "to": "",
                "compensationDetails": {
                  "inHand": "",
                  "rsus": "",
                  "bonus": "",
                  "otherBenefits": ""
                },
                "noticePeriod": "",
                "description": [""]
              }
            ],
            "skills": [
              {
                "skill": "",
                "yearsOfExperience": ""
              }
            ],
            "links": []
          }
        }

        Resume text:
        """
        
        try:
            response = self.model.generate_content(prompt + text)
            # Extract JSON from response
            response_text = response.text
            # Find the JSON part of the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        except Exception as e:
            raise Exception(f"Error parsing resume with Gemini API: {str(e)}")

# Initialize parser with API key
api_key = 'xxx'
if not api_key:
    raise Exception("GOOGLE_API_KEY environment variable not set")
parser = ResumeParser(api_key)

@app.route('/parse-resume', methods=['POST'])
def parse_resume():
    """
    Endpoint to parse a resume file
    Expects a file upload with key 'resume'
    """
    try:
        # Check if a file was uploaded
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume']
        
        # Check if a file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Extract text from file
            text = parser.extract_text(file_path)
            
            # Parse resume
            parsed_data = parser.parse_resume(text)
            
            # Return parsed data
            return jsonify({
                'status': 'success',
                'data': parsed_data
            })
            
        finally:
            # Clean up - remove uploaded file
            os.remove(file_path)
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)