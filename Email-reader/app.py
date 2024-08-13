
# Importing libraries
import imaplib, email
import io
import os
import boto3
import base64
import json
import uuid
import logging
import pandas as pd
from PIL import Image
from io import BytesIO
from botocore.exceptions import ClientError
from datetime import datetime
from astrapy.db import AstraDB, AstraDBCollection
from dotenv import load_dotenv

load_dotenv()

MAIL_PWD = os.getenv("MAIL_PWD")
USER = os.getenv("USER")
IMAP_URL = os.getenv("IMAP_URL")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
FETCH_GMAIL_ADDRESSES = os.getenv("FROM_GMAIL_ADDRESSES")
IMAGE_MAX_SIZE = 1000 # px

ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_ENDPOINT")
ASTRA_DB_REGION = os.getenv("ASTRA_DB_REGION")
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

astra_db = AstraDB(
    token=ASTRA_DB_TOKEN,
    api_endpoint=ASTRA_DB_ENDPOINT
)

print("Getting AstraDB collection for events ...")
stages_log_collection = AstraDBCollection(
    collection_name="stages_log", 
    astra_db=astra_db
)
print("Getting AstraDB collection for vectors ...")
email_images_collection = AstraDBCollection(
    collection_name="email_images", 
    astra_db=astra_db
)
print("Getting AstraDB collection catalog ...")
catalog_collection = AstraDBCollection(
    collection_name="watches_catalog", 
    astra_db=astra_db
)

#######################
# Set AWS credentials #
#######################

# require aws_cli
s3_client = boto3.client('s3')
bedrock_runtime = boto3.client('bedrock-runtime')

#######################################
# Methods to call LLMs (Claude/Titan) #
#######################################
def claude_description(image):
    # Variables for Bedrock API
    category = 'wrist watches'
    modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
    contentType = 'application/json'
    accept = 'application/json'
    
    prompt = """
    Identify the following product in the image provided.
    Product Category: {product_category}
    
    Return an enhanced description of the product based on the image for better search results.
    Do not include any specific details that can not be confirmed from the image such as the quality of materials, other color options, or exact measurements.
    """

    # Messages
    messages = [
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "source": {
              "type": "base64",
              "media_type": "image/jpeg",
              "data": image
            }
          },
          {
            "type": "text",
            "text": prompt.format(product_category=category)
          }
        ]
      }
    ]

    # Body
    claude_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": messages
    })

    # Run Bedrock API to invoke Claude 3 model
    claude_response = bedrock_runtime.invoke_model(
      modelId=modelId,
      contentType=contentType,
      accept=accept,
      body=claude_body
    )

    claude_response_body = json.loads(claude_response.get('body').read())
    return claude_response_body['content'][0]['text']

def generate_titan_embedding(input_text, input_image):
    # Variables for Bedrock API
    embedding_output_length = 1024
    embedding_model_id = "amazon.titan-embed-image-v1"
    contentType = 'application/json'
    accept = 'application/json'
    
    titan_body = json.dumps({
        "inputText": input_text,
        "inputImage": input_image,
        "embeddingConfig": {
            "outputEmbeddingLength": embedding_output_length
        }
    })

    titan_response = bedrock_runtime.invoke_model(
        modelId=embedding_model_id,
        contentType=contentType,
        accept=accept,
        body=titan_body
    )

    final_response = json.loads(titan_response.get('body').read())
    return final_response['embedding']

############################
# Methods to manage images #
############################
def resize_image(img_data, maxwidth, maxheight):
    image = Image.open(BytesIO(img_data))
    width, height = image.size
    if width > IMAGE_MAX_SIZE or height > IMAGE_MAX_SIZE:
        ratio = min(maxwidth/width, maxheight/height)
        newsize = int(width*ratio), int(height*ratio)
        image = image.resize(newsize)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

####################################################
# Methods to Get Emails and images and stored them #
####################################################
def email_connection(inbox):
    print('Logging in again')
    mail = imaplib.IMAP4_SSL(IMAP_URL) 
    mail.login(USER, MAIL_PWD) 
    mail.select(inbox) # Take the readonly=True for prod
    return mail

# Function to get email content part i.e its body part
def get_body(id, mail):
    """Get the body of the email"""
    result = []
    try:
        res, mail_data = mail.fetch(id, '(RFC822)')
        email_message_instance = email.message_from_bytes(mail_data[0][1])
        for part in email_message_instance.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue
            if part.get('Content-ID') is not None:
                img_data = part.get_payload(decode=True)
                image = resize_image(img_data, IMAGE_MAX_SIZE, IMAGE_MAX_SIZE)
                result = [image, None]
                return result
            fileName = part.get_filename()
            print('Attachment being processed ...')
            if bool(fileName):
                img_data = part.get_payload(decode=True)
                image = resize_image(img_data, IMAGE_MAX_SIZE, IMAGE_MAX_SIZE)    
                result = [image, None]
                return result
            else:
                continue
    except Exception as error: 
        errors.append(error)
        result = [None, error]
        return result
    return result

def get_email_images():
    images = []
    mail_conn = email_connection("Inbox")
    for address in FETCH_GMAIL_ADDRESSES:
        result, result_bytes = mail_conn.search(None, f'(FROM "{address}" UNSEEN)')
        email_results = result_bytes[0].split()
    
        for msg in email_results:
            trans_id = str(uuid.uuid4())
            start_time = str(datetime.now())
            result_obj = get_body(msg, mail_conn)
            if result_obj is None:
                print("Desastre")
                break
            if result_obj and str(result_obj[1]) == "socket error: EOF occurred in violation of protocol (_ssl.c:2426)":
                print("Intentaré cerrar conexión")
                mail_conn.logout()
                mail_conn = email_connection("Inbox")
                result_obj = get_body(msg)
            if result_obj and result_obj[0]:
                images.append([trans_id, address, result_obj[0], 'url', 'description', [1.0]*1024, ['']*5, start_time, str(datetime.now()), 0, 0]) 
        print(f'Correos checados {len(email_results)}')
    return images

### Store the image in s3
def s3store_image(image):
    BUCKET_NAME = "aws-speed-date"
    s3_path = "user/"
    s3_obj_path = "{img_path}{filename}.jpg".format(img_path = s3_path, filename = image[0])
    full_obj_path = "{endpoint}{img_path}{filename}.jpg".format(endpoint=S3_ENDPOINT, img_path = s3_path, filename = image[0])
    image[3] = full_obj_path
    print(full_obj_path)
    try:
        response = s3_client.put_object(Body=image[2], Bucket=BUCKET_NAME, Key=s3_obj_path, Tagging=image[1])
    except ClientError as e:
        logging.error(e)
        return False
    return response

################################
# Methods to write to Astra DB #
################################
def write_to_stages(row, event_label, startlabel, endlabel):
    stages_log_collection.insert_one({
        "stage": event_label,
        "start": row[startlabel], 
        "end": row[endlabel],
        "email_address": row['email'],
        "trans_id": row['trans_id'],
        "image_path": row['s3_url'],})

def write_to_email_images(row):
    row['end_astra_store'] = str(datetime.now())
    email_images_collection.insert_one({
        "email": row['email'],
        "trans_id": row['trans_id'],
        "s3_url": row['s3_url'],
        "description": row['description'],
        "start_dwnld": row['start_dwnld'],
        "end_dwnld": row['start_dwnld'],
        "start_s3_store": row['start_s3_store'],
        "end_s3_store": row['end_s3_store'],
        "start_get_desc": row['start_get_desc'],
        "end_get_desc": row['end_get_desc'],
        "start_embedding": row['start_embedding'],
        "end_embedding": row['end_embedding'],
        "start_vsearch": row['start_vsearch'],
        "end_vsearch": row['end_vsearch'],
        "start_astra_store": row['start_astra_store'],
        "end_astra_store": row['end_astra_store'],
        "vsearch_results": row['vsearch_results'],
        "$vector": row['embedding']
        })

###############
# Main script #
###############

errors = []

print("Recuperando correos ...")
user_images = get_email_images()
# Store download stage in Astra
for item in user_images: 
    s3_path = "user/"
    full_obj_path = "{endpoint}{img_path}{filename}.jpg".format(endpoint=S3_ENDPOINT, img_path = s3_path, filename = item[0])
    item[3] = full_obj_path
    stages_log_collection.insert_one({
        "stage": "Image downloaded",
        "start": item[7], 
        "end": item[8],
        "email_address": item[1],
        "trans_id": item[0],
        "image_path": item[3],})
    
print(f'Imagenes descargadas {len(user_images)}')

print("Guardando en S3 ...")

for img in user_images:
    img[9] = str(datetime.now())
    result = s3store_image(img)
    img[10] = str(datetime.now())

# add the missing fields to store start and end timestamps
user_images = [x + ['','','','','','','',''] for x in user_images]
# Create the pandas.DataFrame
df = pd.DataFrame(user_images, columns=['trans_id', 'email', 'img_obj', 's3_url', 'description', 'embedding', 'vsearch_results', 
                                        'start_dwnld', 'end_dwnld', 
                           'start_s3_store', 'end_s3_store', 'start_get_desc', 'end_get_desc', 'start_embedding', 'end_embedding', 
                           'start_vsearch', 'end_vsearch', 'start_astra_store', 'end_astra_store'])

### Store download and s3 storage events in Astra

if not df.empty: 
    for index, row in df.iterrows():
        write_to_stages(row, "Image stored in S3", 'start_s3_store', 'end_s3_store')

### Get Claude descriptions and Titan embeddings and ... Vector Search ###

if not df.empty: 
    for index, row in df.iterrows():
        print("Getting Claude3 description for " + row['trans_id'] + " ...")
        df.loc[index, 'start_get_desc'] = str(datetime.now())
        # b64 representation of the image for Claude
        b64_img_data = base64.b64encode(row['img_obj']).decode("utf-8")
        img_description = claude_description(b64_img_data)
        df.loc[index, 'description'] = img_description
        df.loc[index, 'end_get_desc'] = str(datetime.now())
        write_to_stages(row, "Got Claude3 description", 'start_get_desc', 'end_get_desc')

        ### Get embeddings ###
        print("Getting Titan embedding " + row['trans_id'] + " ...")
        df.loc[index, 'start_embedding'] = str(datetime.now())
        embedding = generate_titan_embedding(row['description'], b64_img_data)
        df.loc[index, 'embedding'] = embedding
        df.loc[index, 'end_embedding'] = str(datetime.now())
        write_to_stages(row, "Got Titan embedding", 'start_embedding', 'end_embedding')

        ### Get Vector Search ###
        print("Getting Similar vectors " + row['trans_id'] + " ...")
        df.loc[index, 'start_vsearch'] = str(datetime.now())
        query_vector = df.loc[index]['embedding']
        documents = catalog_collection.vector_find(
            query_vector,
            limit=5,
            fields=["_id", "file_path", "brand", "product_name", "$vector"],  # remember the dollar sign (reserved name)
            include_similarity=True,
        )
        for document in documents:
            del document['$vector']
            similarity = document['$similarity']
            del document['$similarity']
            document['similarity'] = similarity
            just_path = document['file_path']
            document['file_path'] = f"{S3_ENDPOINT}{just_path}"
        df.loc[index, 'vsearch_results'] = documents
        df.loc[index, 'end_vsearch'] = str(datetime.now())
        write_to_stages(row, "Vector Search done", 'start_vsearch', 'end_vsearch')

        ### Add new completed row to AstraDb ###
        print("Storing in Astra")
        df.loc[index, 'start_astra_store'] = str(datetime.now())
        write_to_email_images(row)
        
print("Finish!")
