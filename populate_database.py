#load documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
#split documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
#creating databases
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

import argparse
import os
import shutil


DATA_PATH = "data"
CHROMA_PATH = "chroma"



def main():

    #check if database should be cleared (using --clear flag)
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()
    
    #create/update data store
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)



#load documents
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# documents = load_documents()
# print(documents[0])
# print()
# print()



#split docs
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

# documents = load_documents()
# chunks = split_documents(documents)
# print(chunks[0])

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    #calculate page ids
    chunks_with_ids = calculate_chunk_ids(chunks)

    #add/update docs
    existing_items = db.get(include=[]) #ids are included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    #only add docs that dont exist in the db
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids = new_chunk_ids)
    else:
        print("No new documents to add")


#def calculate chunk ID
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        #If page ID is the same as the last one, increment the index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        #calculate chunkid
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        #add to the page metadata
        chunk.metadata["id"] = chunk_id
    return chunks



def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)



if __name__ == "__main__":
    main()