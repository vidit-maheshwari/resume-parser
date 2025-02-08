import pdfplumber
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import json
from groq import Groq
import logging
import traceback
import tiktoken

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def count_tokens(text):
    """Estimate token count using GPT-2 tokenizer"""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

class ResumeParser:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.MAX_TOKENS = 2000  # Conservative limit for input text

    def extract_text_from_pdf(self, file_path):
        """Extract text from a PDF file."""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF.")

            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise RuntimeError(f"Failed to extract text: {str(e)}")

    def clean_json_string(self, text):
        """Clean and validate JSON string."""
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = text[start:end]
            parsed = json.loads(json_str)
            return json.dumps(parsed)
        except Exception as e:
            logger.error(f"JSON cleaning error: {str(e)}")
            logger.error(f"Problematic text: {text}")
            raise ValueError(f"Invalid JSON structure: {str(e)}")

    def truncate_to_token_limit(self, text, max_tokens):
        """Truncate text to approximate token limit"""
        while count_tokens(text) > max_tokens:
            text = text[:int(len(text) * 0.9)]  # Reduce by 10% each time
        return text

    def parse_resume(self, file_path):
        """Parse resume by extracting text and sending it to the LLM."""
        try:
            # Step 1: Extract text from PDF
            extracted_text = self.extract_text_from_pdf(file_path)
            logger.info(f"Extracted text length: {len(extracted_text)} characters")

            # Step 2: Truncate content to fit within token limits
            truncated_content = self.truncate_to_token_limit(extracted_text, self.MAX_TOKENS)
            logger.info(f"Truncated text length: {len(truncated_content)} characters")

            # Step 3: Prepare prompt
            prompt = """Extract resume information from the following text. Return a JSON object with key details:
{
  "profile": {
    "location": {"current": "", "relocation": ""},
    "education": {"college": "", "degree": "", "stream": ""},
    "professionalExperience": [
      {
        "company": "",
        "from": "",
        "to": "",
        "description": [""]
      }
    ],
    "skills": [{"skill": "", "yearsOfExperience": ""}]
  }
}

Resume Text: """

            logger.info("Sending request to Groq API")

            # Step 4: Send to LLM
            completion = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": f"{prompt}\n{truncated_content}"
                }],
                model="deepseek-r1-distill-llama-70b",
                temperature=0.0,
                max_tokens=4000
            )

            logger.info("Received response from Groq API")
            response_text = completion.choices[0].message.content
            logger.debug(f"Raw API response: {response_text[:500]}...")

            # Step 5: Extract JSON response
            clean_json = self.clean_json_string(response_text)
            return json.loads(clean_json)

        except Exception as e:
            logger.error(f"Error in parse_resume: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Error parsing resume: {str(e)}")

# Replace with environment variable in production
api_key = 'xxxx'  # Replace with your actual API key
parser = ResumeParser(api_key)

@app.route('/parse-resume', methods=['POST'])
def parse_resume():
    """API endpoint to parse resumes."""
    try:
        logger.info("Received parse-resume request")

        if 'resume' not in request.files:
            logger.error("No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['resume']
        logger.info(f"Received file: {file.filename}")

        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': f'File type not allowed. Allowed types: {ALLOWED_EXTENSIONS}'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.info(f"File saved to: {file_path}")

        try:
            parsed_data = parser.parse_resume(file_path)
            logger.info("Successfully parsed resume")
            return jsonify({
                'status': 'success',
                'data': parsed_data
            })
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)
