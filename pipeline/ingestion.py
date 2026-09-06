from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def is_scanned_pdf(file) -> bool:
    """Check if PDF is scanned by checking if text extraction returns empty."""
    reader = PdfReader(file)
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            return False
    return True


def load_pdf(file) -> str:
    """Extract text from a digital PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
