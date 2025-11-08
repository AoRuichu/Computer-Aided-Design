from google import genai
from google.genai import types
from utils.save_response import writeMd
from dotenv import load_dotenv
from pydantic import BaseModel, create_model, Field
import yaml
from toolify import Args, TD3Runner, CircuitEvaluator

env_path = "./local.env"
load_dotenv(dotenv_path=env_path)

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()


## Complete the code!

TODO: ...