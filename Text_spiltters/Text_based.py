from langchain_text_splitters import RecursiveCharacterTextSplitter
text = '''My name is Jatin and i am 22 years old. I live in Hyderabad and I love playing football
'''
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 8,
    chunk_overlap = 0,
)
chunks = splitter.split_text(text)
print(len(chunks))
print(chunks)