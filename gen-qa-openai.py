import itertools
import argparse
import os
import sys
from time import sleep
from typing import Any, List, Dict
import openai
from tqdm import tqdm
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
parser.add_argument('--short_mode', action='store_true', help="Process very few documents for demo only")
parser.add_argument('--skip_question', action='store_true', help="Do not prompt for a question (useful if just loading data)")
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
mySplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)

if args.load_data:
    #
    if args.short_mode:
        numDocs = 3
        limiter = range(numDocs)
    else:
        numDocs = 700 # here we're cheating
        limiter = itertools.count()
    #
    data = load_dataset('jamescalam/youtube-transcriptions', split='train')
    # collate texts into documents by video_id, one doc per video
    groups = itertools.groupby(data, key=lambda entry: entry['video_id'])
    lcDocuments = []
    print('Loading docs')
    with tqdm(total=numDocs) as prog:
        for (_, grouper), _ in zip(groups, limiter):
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
            a=1
            lcDocuments.append(Document(
                page_content=fulltext,
                metadata=metadata,
            ))

            splitDocs = mySplitter.transform_documents(lcDocuments)

            myCassandraVStore.add_documents(splitDocs)
            prog.update(1)

if not args.skip_question:
    # Now we search
    query = f"Which training method should I use for sentence transformers when " + \
            f"I only have pairs of related sentences?"

    print(f'Query? (enter to use default)\n\nDefault:\n{query}\n\n> ', end='')
    query = sys.stdin.readline().strip() or query

    response = index.query(query)

    print(f'Answer: {response}')
