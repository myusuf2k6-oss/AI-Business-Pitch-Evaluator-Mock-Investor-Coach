"""Tests for the document parsing module document_parser.py."""

from unittest.mock import MagicMock, patch
import pytest
from app.backend.document_parser import parse_pdf, parse_docx, extract_text, clean_text
from app.backend.exceptions import DocumentParsingError

def test_clean_text():
    """Verify that clean_text removes extra whitespace and formats lines."""
    input_text = "  Hello   World!  \n\n  This is   a test.  "
    expected = "Hello World! This is a test."
    assert clean_text(input_text) == expected

@patch("app.backend.document_parser.PdfReader")
def test_parse_pdf_success(mock_pdf_reader):
    """Verify successful PDF parsing using mocked PdfReader."""
    # Setup mock page objects
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page 1 Content"
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page 2 Content"
    
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page1, mock_page2]
    mock_pdf_reader.return_value = mock_reader
    
    result = parse_pdf(b"dummy_pdf_bytes")
    assert result == "Page 1 Content Page 2 Content"

@patch("app.backend.document_parser.PdfReader")
def test_parse_pdf_empty(mock_pdf_reader):
    """Verify that an empty PDF raises a DocumentParsingError."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "   "
    
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]
    mock_pdf_reader.return_value = mock_reader
    
    with pytest.raises(DocumentParsingError, match="empty or contains only images"):
        parse_pdf(b"dummy_bytes")

@patch("app.backend.document_parser.docx.Document")
def test_parse_docx_success(mock_docx_document):
    """Verify successful DOCX parsing using mocked Document object."""
    mock_para1 = MagicMock()
    mock_para1.text = "Paragraph 1"
    mock_para2 = MagicMock()
    mock_para2.text = "Paragraph 2"
    
    mock_cell = MagicMock()
    mock_cell.text = "Table Cell Content"
    mock_row = MagicMock()
    mock_row.cells = [mock_cell]
    mock_table = MagicMock()
    mock_table.rows = [mock_row]
    
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para1, mock_para2]
    mock_doc.tables = [mock_table]
    mock_docx_document.return_value = mock_doc
    
    result = parse_docx(b"dummy_docx_bytes")
    assert "Paragraph 1" in result
    assert "Paragraph 2" in result
    assert "Table Cell Content" in result

def test_extract_text_unsupported_extension():
    """Verify unsupported files raise a DocumentParsingError."""
    with pytest.raises(DocumentParsingError, match="Unsupported file format"):
        extract_text(b"data", "pitch.txt")

@patch("app.backend.document_parser.parse_pdf")
def test_extract_text_routing_pdf(mock_parse_pdf):
    """Verify extract_text routes PDF files correctly."""
    mock_parse_pdf.return_value = "Parsed PDF text"
    result = extract_text(b"bytes", "presentation.pdf")
    assert result == "Parsed PDF text"
    mock_parse_pdf.assert_called_once_with(b"bytes")

@patch("app.backend.document_parser.parse_docx")
def test_extract_text_routing_docx(mock_parse_docx):
    """Verify extract_text routes DOCX files correctly."""
    mock_parse_docx.return_value = "Parsed DOCX text"
    result = extract_text(b"bytes", "proposal.docx")
    assert result == "Parsed DOCX text"
    mock_parse_docx.assert_called_once_with(b"bytes")
