import os
from unittest import loader
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain_community.document_loaders import DirectoryLoader




def main():
    loader = DirectoryLoader("data/pdf/", glob="**/*.pdf", loader_cls=PyPDFLoader)

    documents = loader.load()
    print(documents)


    
    


if __name__ == "__main__":
    main()
