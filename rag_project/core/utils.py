import re

def clean_text(text: str) -> str:
    """
    Cleans extracted text by:
    - Replacing newlines with spaces
    - Replacing multiple spaces with a single space
    - Stripping leading and trailing whitespace
    """

    if not text:
        return ""

    # Replace newlines with spaces
    text = text.replace("\n", " ")

    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text
#dividing text into chunks with overlap
#Each chunk overlaps the previous one by 50 characters → preserves context.
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks.

    Args:
        text (str): The input text
        chunk_size (int): Size of each chunk
        overlap (int): Number of overlapping characters between chunks

    Returns:
        List[str]: List of text chunks
    """

    chunks = []

    if not text:
        return chunks

    start = 0
    text_length = len(text)

    # Move through the text
    while start < text_length:
        end = start + chunk_size

        chunk = text[start:end]
        chunks.append(chunk)

        # Move start forward by (chunk_size - overlap)
        start = start + (chunk_size - overlap)

    return chunks
