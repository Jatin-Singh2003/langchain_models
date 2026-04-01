from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
text = '''
def decode_string(s):
    stack = []
    curr_str = ""
    curr_num = 0
    
    for char in s:
        if char.isdigit():
            curr_num = curr_num * 10 + int(char)
        elif char == '[':
            stack.append((curr_str, curr_num))
            curr_str, curr_num = "", 0
        elif char == ']':
            prev_str, num = stack.pop()
            curr_str = prev_str + num * curr_str
        else:
            curr_str += char
    
    return curr_str'''
splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 30,
    chunk_overlap = 0
)
chunks = splitter.split_text(text)
print(f'length of chunks is : {len(chunks)}')
print(chunks)