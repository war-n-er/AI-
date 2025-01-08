import boto3
from botocore.exceptions import NoCredentialsError
# nnw
# AWS credentials and S3 bucket configuration
ACCESS_KEY = 'YOUR_AWS_ACCESS_KEY'
SECRET_KEY = 'YOUR_AWS_SECRET_KEY'
bucket_name = 'your-s3-bucket-name'
file_path = 'image_classifier.h5'
s3_key = 'models/image_classifier.h5'

# Upload model to S3
def upload_to_s3(file_path, bucket_name, s3_key):
    try:
        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
        s3.upload_file(file_path, bucket_name, s3_key)
        print(f'Successfully uploaded {file_path} to {bucket_name}/{s3_key}')
    except NoCredentialsError:
        print('Credentials not available.')
# nnw
# MLOps, AWS S3, model storage, and cloud-based deployment
upload_to_s3(file_path, bucket_name, s3_key)
