from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- ingest (same as before) ---
reader = PdfReader("FinalProjectTemplates.pdf")
text = ""
for page in reader.pages:
    text = text + page.extract_text()

# --- chunk (the new part) ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(text)

print(f"Number of chunks: {len(chunks)}")
print("---- first chunk ----")
print(chunks[0])
