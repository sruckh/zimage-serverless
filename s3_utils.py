import os
import boto3
from botocore.client import Config

def upload_image_to_s3(file_path, object_name, expiration=3600):
    """
    Uploads an image to an S3-compatible bucket (e.g., Backblaze B2)
    and returns a pre-signed URL.
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
        # Upload the file
        s3.upload_file(file_path, bucket_name, object_name, ExtraArgs={'ContentType': 'image/png'})
        
        # Generate a pre-signed URL for the uploaded object
        response = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_name},
            ExpiresIn=expiration
        )
        return response
    except Exception as e:
        print(f"Error uploading to S3 or generating pre-signed URL: {e}")
        raise e
