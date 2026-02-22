import os
import boto3
from botocore.client import Config

def upload_image_to_s3(file_path, object_name):
    """
    Uploads an image to an S3-compatible bucket (e.g., Backblaze B2).
    """
    endpoint_url = os.environ.get("S3_ENDPOINT_URL")
    access_key = os.environ.get("S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("S3_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("S3_BUCKET_NAME")

    if not all([endpoint_url, access_key, secret_key, bucket_name]):
        raise ValueError("Missing S3 configuration environment variables.")

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4')
    )

    try:
        s3.upload_file(file_path, bucket_name, object_name, ExtraArgs={'ContentType': 'image/jpeg'})
        # Construct the URL. Note: This assumes the bucket is public or you have a specific URL format.
        # For B2, it's often https://<bucket>.<endpoint>/<object> or similar.
        # A more robust way might be to generate a presigned URL if it's private, 
        # but usually for serverless outputs, we return a direct link if public.
        # We'll return a simple concatenation or the user can configure the base URL.
        base_url = os.environ.get("S3_BASE_URL", f"{endpoint_url}/{bucket_name}")
        return f"{base_url}/{object_name}"
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise e
