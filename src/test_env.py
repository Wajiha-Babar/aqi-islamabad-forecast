import os
from dotenv import load_dotenv

load_dotenv()

print("API KEY:", os.getenv("HOPSWORKS_API_KEY"))
print("PROJECT:", os.getenv("HOPSWORKS_PROJECT"))
