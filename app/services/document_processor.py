"""
Document Processor Service
Handles text chunking using LangChain's RecursiveCharacterTextSplitter
and PDF extraction
"""
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError(
            "LangChain is required. Please install it with: pip install langchain"
        )
from typing import List, Dict
import logging
import io

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("pypdf not available. PDF processing will be disabled.")

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Document processor service using LangChain's RecursiveCharacterTextSplitter
    
    Handles text chunking and PDF extraction for vector store ingestion.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """
        Initialize document processor
        
        Args:
            chunk_size: Maximum size of chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            separators: List of separators to use for splitting
        """
        # Default separators for recursive splitting
        if separators is None:
            separators = [
                "\n\n",  # Paragraphs
                "\n",    # Lines
                ". ",    # Sentences
                " ",     # Words
                ""       # Characters
            ]
        
        # Initialize RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
        
        logger.info(
            f"Document processor initialized: "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )
    
    def extract_text_from_file(self, file_bytes: bytes, filename: str) -> str:
        """
        Extract text from file (PDF or text file)
        
        Args:
            file_bytes: File content as bytes
            filename: Name of the file
            
        Returns:
            str: Extracted text from file
        """
        if filename.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_bytes)
        elif filename.lower().endswith(('.txt', '.text')):
            try:
                text = file_bytes.decode('utf-8')
                if not text.strip():
                    raise ValueError("Text file is empty.")
                return text
            except UnicodeDecodeError:
                for encoding in ['latin-1', 'cp1252']:
                    try:
                        text = file_bytes.decode(encoding)
                        if text.strip():
                            return text
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not decode text file. Please ensure it's a valid text file.")
        else:
            raise ValueError(f"Unsupported file type: {filename}. Supported types: .txt, .pdf")
    
    def extract_text_from_pdf(self, pdf_file: bytes) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_file: PDF file as bytes
            
        Returns:
            str: Extracted text from PDF
        """
        if not PDF_AVAILABLE:
            raise ValueError("PDF processing is not available. Please install pypdf.")
        
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf_file))
            
            text_parts = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
            
            full_text = "\n\n".join(text_parts)
            
            if not full_text.strip():
                raise ValueError("No text could be extracted from the PDF file.")
            
            logger.info(f"Extracted {len(full_text)} characters from PDF ({len(pdf_reader.pages)} pages)")
            return full_text
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    def split_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks using RecursiveCharacterTextSplitter
        
        Args:
            text: Input text to split
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List[Dict]: List of text chunks with metadata
        """
        try:
            chunks = self.text_splitter.split_text(text)
            
            chunk_documents = []
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "text": chunk,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                
                if metadata:
                    chunk_doc.update(metadata)
                
                chunk_documents.append(chunk_doc)
            
            logger.info(
                f"Split text into {len(chunks)} chunks "
                f"(avg size: {sum(len(c) for c in chunks) // len(chunks) if chunks else 0} chars)"
            )
            
            return chunk_documents
            
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            raise
    
    def process_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Process plain text: split into chunks
        
        Args:
            text: Input text
            metadata: Optional metadata for chunks
            
        Returns:
            List[Dict]: List of processed chunks
        """
        return self.split_text(text, metadata)
    
    def process_pdf(self, pdf_file: bytes, metadata: Dict = None) -> List[Dict]:
        """
        Process PDF file: extract text and split into chunks
        
        Args:
            pdf_file: PDF file as bytes
            metadata: Optional metadata for chunks
            
        Returns:
            List[Dict]: List of processed chunks
        """
        text = self.extract_text_from_pdf(pdf_file)
        
        if metadata is None:
            metadata = {}
        metadata["source_type"] = "pdf"
        
        return self.split_text(text, metadata)
    
    def process_documents(
        self,
        texts: List[str] = None,
        pdf_files: List[bytes] = None,
        metadata_list: List[Dict] = None
    ) -> List[Dict]:
        """
        Process multiple documents (texts and/or PDFs)
        
        Args:
            texts: List of text strings
            pdf_files: List of PDF files as bytes
            metadata_list: Optional list of metadata dicts for each document
            
        Returns:
            List[Dict]: List of all processed chunks from all documents
        """
        all_chunks = []
        
        if texts:
            for i, text in enumerate(texts):
                metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
                metadata["document_index"] = i
                metadata["source_type"] = "text"
                chunks = self.process_text(text, metadata)
                all_chunks.extend(chunks)
        
        # Process PDF documents
        if pdf_files:
            start_idx = len(texts) if texts else 0
            for i, pdf_file in enumerate(pdf_files):
                metadata = metadata_list[start_idx + i] if metadata_list and (start_idx + i) < len(metadata_list) else {}
                metadata["document_index"] = start_idx + i
                chunks = self.process_pdf(pdf_file, metadata)
                all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(all_chunks)} total chunks from all documents")
        return all_chunks

