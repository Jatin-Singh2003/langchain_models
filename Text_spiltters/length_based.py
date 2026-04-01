from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
text = '''The quiet hum of the city faded as the early morning mist settled over the narrow streets. A lone bicycle leaned against a weathered brick wall, its tires slightly deflated but still holding the promise of movement. Somewhere in the distance, a vendor arranged fresh fruits in neat rows, carefully aligning oranges and apples as if symmetry itself could attract customers.

Inside a small apartment above the street, a programmer stared at a glowing screen, lost in a maze of logic and edge cases. The code compiled successfully, but something felt off—an invisible bug lurking beneath layers of abstraction. Debugging, much like solving a puzzle, required patience and a willingness to question every assumption.

Across the room, a notebook lay open with hastily scribbled ideas: distributed systems, caching strategies, and thoughts about scaling applications efficiently. The concept of time seemed fluid here; minutes stretched into hours as focus deepened. Outside, the city began to awaken—footsteps, distant chatter, and the occasional honk breaking the calm.

By noon, the mist had vanished, replaced by a sharp clarity that revealed every detail of the surroundings. The bicycle was gone, the fruit stand half-empty, and the day fully in motion. Yet inside, the programmer remained in the same position, chasing the elegant simplicity of a perfect solution.

'''
splitter = CharacterTextSplitter(
    chunk_size = 15,
    chunk_overlap = 0,
    separator= ''
)

loader = PyPDFLoader('/Users/macmini/Downloads/Suchita Resume 2025.pdf')
docs = loader.load()
res1 = splitter.split_documents(docs)
# print(res1[0])
print(res1[2].page_content)