from dotenv import load_dotenv
import os

load_dotenv()

NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
os.environ['NVIDIA_API_KEY'] = NVIDIA_API_KEY