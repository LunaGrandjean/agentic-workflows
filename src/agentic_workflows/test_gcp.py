import os

from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud import storage


# Charge les variables contenues dans le fichier .env
load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")


print("=== Configuration GCP ===")
print(f"Project ID : {GCP_PROJECT_ID}")
print(f"Region     : {GCP_REGION}")
print(f"Bucket     : {GCP_BUCKET_NAME}")


# 1. Test Vertex AI
print("\n=== Test Vertex AI ===")

aiplatform.init(
    project=GCP_PROJECT_ID,
    location=GCP_REGION,
)

print("Vertex AI initialisé avec succès")


# 2. Test Google Cloud Storage
print("\n=== Test Google Cloud Storage ===")

client = storage.Client(project=GCP_PROJECT_ID)
bucket = client.bucket(GCP_BUCKET_NAME)

print(f"Accès au bucket : gs://{bucket.name}")


# 3. Afficher le contenu du bucket
print("\n=== Contenu du bucket ===")

blobs = list(client.list_blobs(GCP_BUCKET_NAME))

if blobs:
    for blob in blobs:
        print(f" - {blob.name}")
else:
    print("Le bucket est vide.")

print("\nConfiguration GCP fonctionnelle !")