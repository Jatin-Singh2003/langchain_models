from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('/Users/macmini/Downloads/Suchita Resume 2025.pdf')
docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)
print(docs[1].page_content)