

from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import json
from groq import Groq
import base64
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
        """Parse resume by sending the file directly to the LLM."""
        try:
            # Read the file as binary
            with open(file_path, 'rb') as file:
                file_content = file.read()
            
            # Convert binary to base64 for text representation
            file_base64 = base64.b64encode(file_content).decode('utf-8')
            
            logger.info(f"Original file size after base64 encoding: {len(file_base64)} characters")
            
            # Truncate content to fit within token limits
            truncated_content = self.truncate_to_token_limit(file_base64, self.MAX_TOKENS)
            logger.info(f"Truncated file size: {len(truncated_content)} characters")

            prompt = """Extract resume information from this base64 encoded document. Return a JSON object with key information:
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

Document: """

            logger.info("Sending request to Groq API")
            
            completion = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": f"{prompt}\n{truncated_content}"
                }],
                model="deepseek-r1-distill-llama-70b",
                temperature=0.,
                max_tokens=4000
            )
            
            logger.info("Received response from Groq API")
            response_text = completion.choices[0].message.content
            logger.debug(f"Raw API response: {response_text[:500]}...")
            
            clean_json = self.clean_json_string(response_text)
            return json.loads(clean_json)
            
        except Exception as e:
            logger.error(f"Error in parse_resume: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Error parsing resume: {str(e)}")

# Replace with environment variable in production
api_key = 'xxx'  # Replace with your actual API key
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