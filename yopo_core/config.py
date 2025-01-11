import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('DEFAULT_API_KEY')
API_BASE = os.getenv('DEFAULT_API_BASE')
MODEL_ID = os.getenv('DEFAULT_MODEL_NAME')

os.environ['OPENAI_API_BASE'] = API_BASE
os.environ['OPENAI_API_KEY'] = API_KEY