from itertools import groupby
import argparse
import os
import sys
from time import sleep
from typing import Any, List, Dict
import openai
from tqdm.auto import tqdm
from datasets import load_dataset

from langchain.schema import Document
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

import cassio
cassio.globals.enableExperimentalVectorSearch()

from db import getSession, createKeyspace

os.environ['OPENAI_API_KEY'] = open('openai.key', 'r').read().splitlines()[0]
embed_model = "text-embedding-ada-002"

parser = argparse.ArgumentParser(description="Retrieve data and generate answers")
parser.add_argument('--load_data', action='store_true', help="Load data if required")
args = parser.parse_args()

#
session = getSession()
createKeyspace(session, 'demo_lc')

llm = OpenAI(temperature=0)
myEmbedding = OpenAIEmbeddings(model=embed_model)

myCassandraVStore = Cassandra(
    embedding=myEmbedding,
    session=session,
    keyspace='demo_lc',
    table_name='youtube_transcriptions',
)

index = VectorStoreIndexWrapper(vectorstore=myCassandraVStore)

if args.load_data:
    data = load_dataset('jamescalam/youtube-transcriptions', split='train')
    # collate texts into documents by video_id, one doc per video
    groups = groupby(data, key=lambda entry: entry['video_id'])
    lcDocuments = []
    for _, grouper in groups:
        chunks = list(grouper)
        fulltext = ' '.join(
            chk['text']
            for chk in chunks
        )
        metadata = {
            k: v
            for k, v in chunks[0].items()
            if k != 'text'
        }
        lcDocuments.append(Document(
            page_content=fulltext,
            metadata=metadata,
        ))

    # # DEMO MODE, we keep 3 videos only
    # lcDocuments = lcDocuments[:3]

    mySplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    splitDocs = mySplitter.transform_documents(lcDocuments)

    myCassandraVStore.add_documents(splitDocs)

# Now we search
query = f"Which training method should I use for sentence transformers when " + \
        f"I only have pairs of related sentences?"

print(f'Query? (enter to use default)\n\nDefault:\n{query}\n\n> ', end='')
query = sys.stdin.readline().strip() or query

response = index.query(query)

print(f'Answer: {response}')
