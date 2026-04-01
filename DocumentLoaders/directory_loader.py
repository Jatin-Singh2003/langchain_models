from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
loader = DirectoryLoader(path='/Users/macmini/Desktop/whitebox', glob="**/*.txt")
#loads all the files present in a directory
#we provide different symbols in glob parameter to identify the files