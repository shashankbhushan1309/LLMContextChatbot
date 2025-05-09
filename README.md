# PDF Question Answering System

This application allows users to upload one or more PDF documents (research papers, product manuals, policies, etc.) and ask questions about their content. The system processes the PDFs, chunks them intelligently, creates embeddings, stores them in a vector database, and answers questions using the google/flan-t5-large language model.

## Features

- Upload and process multiple PDF files
- Intelligent text chunking with overlap
- Document embedding using sentence-transformers
- Vector similarity search with ChromaDB
- LLM-powered question answering with flan-t5-large
- Filter questions by document source
- Modern web interface

## Architecture

The system consists of the following components:

1. **PDF Processor**: Extracts text from PDFs and chunks it intelligently at paragraph boundaries
2. **Embedding Generator**: Creates embeddings for text chunks using sentence-transformers
3. **Vector Database**: Stores embeddings and allows similarity search with ChromaDB
4. **LLM Module**: Answers questions based on retrieved contexts using google/flan-t5-large
5. **Web Interface**: FastAPI backend with HTML/JS frontend

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```
git clone <repository-url>
cd pdf-qa-system
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

### Running the Application

1. Start the server:
```
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

## Usage

1. **Upload PDFs**: Use the file upload form to upload one or more PDF documents.

2. **Ask Questions**: Type your question in the input field. You can optionally:
   - Filter by a specific document
   - Adjust the number of context chunks used for answering

3. **View Answers**: The system will display the answer along with the source documents used to generate it.

## Technical Details

- Chunking Strategy: The system uses intelligent chunking that respects paragraph boundaries while maintaining context
- Embeddings: Uses sentence-transformers to create dense vector representations of text
- Vector Search: ChromaDB provides fast similarity search capabilities
- Language Model: Uses google/flan-t5-large for answer generation
- Web Framework: FastAPI for backend, Jinja2 for templating

## Limitations

- Large PDFs with complex formatting or heavy use of tables/images may not be processed correctly
- The system's answers are limited by the capabilities of the flan-t5-large model
- Processing large documents may require significant memory and compute resources

## Future Improvements

- Add support for more document formats (DOCX, TXT, etc.)
- Implement more advanced chunking strategies
- Add document-level and chunk-level metadata
- Support multi-modal question answering with images
- Implement authentication and user-specific document collections 

## Latest Update

The system now uses Google's Gemini API for advanced question answering. This provides significantly better responses compared to the previous local model approach.

## API Key Setup

To use this system, you need to set up your Gemini API key:

1. Create a Google AI Studio account and get your API key from https://aistudio.google.com/
2. Open the `config.env` file in the project root
3. Replace `your_gemini_api_key_here` with your actual API key
4. Optionally adjust other settings in the config file 