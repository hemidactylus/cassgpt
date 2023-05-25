# Q&A with ChatGPT enriched with youtube transcriptions

**LANGCHAIN VARIANT**

This program uses OpenAI's GPT-3 to generate answers to your questions based on transcriptions of presentations on AI from YouTube.

## Dependencies

- OpenAI API key, stored in file `openai.key`
- Cassandra database supporting vector search. Currently that means you need to build and run 
this branch: https://github.com/datastax/cassandra/tree/cep-vsearch
  - TLDR `git clone`, `ant jar`, `bin/cassandra -f`
  - ⚠️ !Note! This is a prerelease and uses `FLOAT VECTOR[N]` syntax for declaring a vector column.
    This will change in the near future.
- JDK 11.  Exactly 11.
- You will be able to run cqlsh with vector support if you run `bin/cqlsh` from the cassandra source root
- You can install the Python dependencies for cassgpt by running 
`pip install -r requirements.txt` from this source tree.

## Usage
`python gen-qa-openai.py [--load_data [--short_mode]] [--skip_question]`

Specifying `--load_data` will will download the dataset, merge the transcriptions into larger chunks, generate embeddings for each chunk using OpenAI's `text-embedding-ada-002` model, and insert the chunks and embeddings into the database.
This only needs to be done once.
This version does that using LangChain's text splitters, that are rather slow. The splitting into chunks is such that
this data upload step may well last hours.

Once the dataset is loaded, the program will prompt you for a question; it will find the most
relevant context from the transcriptions using Cassandra vector search, and feed the resulting
context + question to OpenAI to generate an answer to your query.

If it's just a functional demo you're after, add `--short_mode` and only the first three Youtube video transcriptions will be indexed.
This means a couple of minutes, but don't expect particularly illuminating answers to your questions. Truncate the table once you're done
if you want to load the full thing (to prevent duplicates).

Assumes Cassandra is running on localhost, hack the source if it's somewhere else.

## Need to start over?
Instead of rebuilding the embeddings from scratch (slow!), dump them from Cassandra and
re-load them into a fresh database.

`python dump.py`
`python load.py`

Also assumes Cassandra is running on localhost.

---
