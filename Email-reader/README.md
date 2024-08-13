# Read images from a Gmail account
App for retrieving images from a gmail inbox, using username and password.

This application is meant to run inside a cron job or as a service, along side the other applications in this same repository.

## What the application does
The app's job is to get an image of a wrist watch inside emails sent to the inbox from certain email address, it will use Bedrock - Claude 3 Sonnet to get a description of the wrist watch, it wiil then use Beddrock Titan G1 to create a multimodal embedding with the actual image and the description (from Claude), and finally store the images retrieved with its generated description and embedding in AstraDB.  

## Environment configuration
The configuration data is taken from a .env file, a sample for this file is provided.