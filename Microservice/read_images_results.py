import asyncio
import os
import astrapy
from astrapy import AsyncCollection
from dotenv import load_dotenv
from flask import Flask

load_dotenv()  # take environment variables from .env.

# This is a DB in Roberto's Org.
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")

my_client = astrapy.DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
my_database = my_client.get_database(ASTRA_DB_API_ENDPOINT)
my_async_database = my_client.get_async_database(ASTRA_DB_API_ENDPOINT)

# Initialize the client and get a "Database" object
# client = DataAPIClient(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
# database = client.get_database(os.environ["ASTRA_DB_API_ENDPOINT"])
print(f"* Database: {my_database.info().name}\n")

# images_collection = my_database.get_collection("email_images")
sync_collection = my_database.get_collection("email_images")


async def read_images():
    async_email_images_collection = AsyncCollection(database=my_async_database, name="email_images")
    search_filter = {"email": {"$exists": True}}
    last_5_images_cursor = (
        async_email_images_collection.find(
            search_filter,
            limit=5,
            sort={"start_dwnld": astrapy.constants.SortDocuments.DESCENDING},))
    docs = [doc async for doc in last_5_images_cursor]
    print("find results 5:", docs)
    return docs


app = Flask(__name__)


@app.route('/')
def hello():
    what_am_i = "This is a python program|" + "read_images_results" + "|||"
    return what_am_i


@app.route('/images')
def get_images():
    json_data = asyncio.run(read_images())
    print("==================")
    print(json_data)
    print("==================")
    return json_data


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8980, debug=True, threaded=True)
