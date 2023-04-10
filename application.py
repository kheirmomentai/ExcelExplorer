import pandas as pd
import pinecone
import openai
from tqdm.auto import tqdm
from time import sleep
from tqdm.auto import tqdm
import datetime
from time import sleep
index_name = 'gpt-4-langchain-docs'
openai.api_key = "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"
embed_model = "text-embedding-ada-002"
# initialize connection to pinecone
pinecone.init(
    api_key="17516b20-b2a9-4a05-80bf-1a610b963582",  # app.pinecone.io (console)
    environment="us-central1-gcp"  # next to API key in console
)


# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='dotproduct'
    )
# connect to index
index = pinecone.GRPCIndex(index_name)
# view index stats
index.describe_index_stats()
# Read the Excel file
file_path = "data1.csv"
df = pd.read_csv(file_path)

# Convert the DataFrame to a string (remove index and header)
excel_string = df.to_string(index=False, header=False)

# Extract header and convert it to a string
header_str = ' '.join(df.columns)

# Convert the DataFrame to a string (remove index and header)
excel_string = df.to_string(index=False, header=False)

# Divide the string into chunks of 5 to 10 rows
rows = excel_string.split('\n')
chunk_size = 15  # You can change this to any number of rows you want per chunk
chunks_text = ['\n'.join([header_str] + rows[i:i+chunk_size]) for i in range(0, len(rows), chunk_size)]

# Convert chunks to the desired format
chunks = []
for i, chunk in enumerate(chunks_text):
    chunk_dict = {
        "id": str(i),  # Generate your own unique ID here
        "text": chunk,
        "chunk": i,
    }
    chunks.append(chunk_dict)


batch_size = 2  # how many embeddings we create and insert at once

for i in tqdm(range(0, len(chunks), batch_size)):
    # find end of batch
    i_end = min(len(chunks), i+batch_size)
    meta_batch = chunks[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass
    embeds = [record['embedding'] for record in res['data']]
    # cleanup metadata
    meta_batch = [{
        'text': x['text'],
        'chunk': x['chunk'],

    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)


from flask import Flask, request, render_template_string

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query = request.form.get("query")

        # Get the embedding for the query
        res = openai.Embedding.create(input=[query], engine=embed_model)

        # Retrieve from Pinecone
        xq = res['data'][0]['embedding']
        res = index.query(xq, top_k=5, include_metadata=True)

        # Get list of retrieved text
        contexts = [item['metadata']['text'] for item in res['matches']]
        augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query

        # System message to 'prime' the model
        primer = f"""You are Q&A bot. A highly intelligent system that answers user questions based on the information provided by the user above
each question. If the information can not be found in the information provided by the user you please answer in general and please do not say that there is no information or not mentioned just say the answer".
"""

        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": primer},
                {"role": "user", "content": augmented_query}
            ]
        )

        answer = res["choices"][0]["message"]["content"]
        return render_template_string("<p>{{ answer }}</p>", answer=answer)

    return render_template_string("""
        <form method="post">
            <label for="query">Enter your query:</label>
            <input type="text" id="query" name="query" required>
            <button type="submit">Submit</button>
        </form>
    """)

if __name__ == "__main__":
    app.run(port=8000)

