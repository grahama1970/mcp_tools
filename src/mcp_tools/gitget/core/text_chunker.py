#!/usr/bin/env python3
"""
Text Chunking Module for Git Repository Analysis

This module provides functionality to break down text into manageable chunks
while preserving section structure, hierarchy, and metadata for efficient
analysis and processing.

This module is part of the Core Layer and should have no dependencies on
Presentation or Integration layers.

Links to documentation:
- tiktoken: https://github.com/openai/tiktoken
- spaCy: https://spacy.io/api/doc

Sample input:
- text: "# Section 1\nContent for section 1\n## Subsection 1.1\nMore content..."
- max_tokens: 500
- min_overlap: 100
- model_name: "gpt-4"

Expected output:
- List of dictionaries containing chunked text with metadata
- Each chunk includes section path, content, and token counts
"""

import os
import re
import datetime
import hashlib
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set, Union
from loguru import logger

# Import required packages
try:
    import tiktoken
    tiktoken_available = True
except ImportError:
    logger.warning("tiktoken not available, falling back to character counting")
    tiktoken_available = False

try:
    import spacy
    spacy_available = True
except ImportError:
    logger.warning("spaCy not available, falling back to regex sentence splitting")
    spacy_available = False

def hash_string(input_string: str) -> str:
    """
    Creates a stable hash for a given string using SHA256.
    
    Args:
        input_string: The string to hash.
        
    Returns:
        A hexadecimal string representation of the hash.
    """
    encoded_string = input_string.encode("utf-8")
    hash_object = hashlib.sha256(encoded_string)
    return hash_object.hexdigest()

class SectionHierarchy:
    """
    Manages a stack representing the current section hierarchy path.
    
    This class tracks the nesting of sections in a document, allowing for
    preservation of section relationships when chunking text.
    """

    def __init__(self):
        """Initialize the section hierarchy stack."""
        # Stack stores tuples: (section_number_str, full_title_str, content_hash_str)
        self.stack: List[Tuple[str, str, str]] = []
        logger.debug("SectionHierarchy initialized.")

    def update(self, section_number: str, section_title: str, section_content: str):
        """
        Updates the hierarchy stack based on a newly encountered section.

        Args:
            section_number: The number string (e.g., "5", "5.2", "5.5.1").
            section_title: The title string (e.g., "Operational Safety Requirements").
            section_content: The content associated with this specific section (used for hashing).
        """
        logger.debug(
            f"Updating hierarchy with section: {section_number=}, {section_title=}"
        )
        # Parse the current section number into a list of ints
        try:
            nums = [int(n) for n in section_number.split(".") if n]
            current_level = len(nums)
            logger.debug(f"Parsed section number parts: {nums=}, {current_level=}")
        except ValueError:
            logger.warning(
                f"Could not parse section number '{section_number}'. Skipping hierarchy update."
            )
            return

        # Pop anything on the stack that isn't an ancestor of this one
        while self.stack:
            prev_number_str, _, _ = self.stack[-1]
            try:
                prev_nums = [int(n) for n in prev_number_str.split(".") if n]
                prev_level = len(prev_nums)
            except ValueError:
                # If somehow invalid, just pop it off
                logger.warning(f"Invalid number on stack '{prev_number_str}', popping.")
                self.stack.pop()
                continue

            # ancestor if prev_level < current_level and prefix matches
            is_ancestor = prev_level < current_level and nums[:prev_level] == prev_nums
            logger.debug(
                f"Is '{prev_number_str}' an ancestor of '{section_number}'? {is_ancestor}"
            )

            if not is_ancestor:
                logger.debug(f"Popping non-ancestor '{prev_number_str}'")
                self.stack.pop()
            else:
                # once we hit a true ancestor, stop popping
                break

        # Now append the current section
        full_title = f"{section_number} {section_title}".strip()
        section_hash = hash_string(full_title + section_content)
        self.stack.append((section_number, full_title, section_hash))
        logger.debug(
            f"Appended to stack: {section_number=}, {full_title=}, hash={section_hash}"
        )
        logger.debug(f"Current hierarchy stack: {[t for _, t, _ in self.stack]}")

    def get_titles(self) -> List[str]:
        """Return all titles in the current hierarchy (ancestors + current)."""
        titles = [title for (_, title, _) in self.stack]
        logger.debug(f"Getting titles: {titles}")
        return titles

    def get_hashes(self) -> List[str]:
        """Return all hashes in the current hierarchy (ancestors + current)."""
        hashes = [h for (_, _, h) in self.stack]
        logger.debug(f"Getting hashes: {hashes}")
        return hashes

    def __str__(self):
        """String representation of the hierarchy."""
        return " -> ".join([title for _, title, _ in self.stack])

def count_tokens_with_tiktoken(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in a string using tiktoken, with fallback to character estimation.
    
    Args:
        text: The text to count tokens for.
        model: The model name to use for token counting.
        
    Returns:
        The number of tokens in the text.
    """
    if tiktoken_available:
        try:
            openai_models = {
                "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k",
                "gpt-4o", "gpt-4-turbo"
            }
            
            # For OpenAI models, use the specific encoding
            if any(model.startswith(m) for m in openai_models):
                encoding = tiktoken.encoding_for_model(model)
            else:
                # For other models, use cl100k_base as default
                encoding = tiktoken.get_encoding("cl100k_base")
                
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens with tiktoken: {e}")
            # Fall back to character estimation
    
    # Fallback: Estimate based on character count
    return int(len(text) / 4)

class TextChunker:
    """
    A class for chunking text while preserving section structure.
    
    This chunker can identify section headers in text, maintain their hierarchical
    relationships, and split the content into chunks of specified token size while
    preserving metadata across chunks.
    """

    def __init__(
        self,
        max_tokens: int = 500,
        min_overlap: int = 100,
        model_name: str = "gpt-4",
        spacy_model: str = "en_core_web_sm",
    ):
        """
        Initialize the TextChunker.
        
        Args:
            max_tokens: Maximum number of tokens per chunk.
            min_overlap: Minimum number of tokens to overlap between chunks.
            model_name: Name of the model for token counting.
            spacy_model: Name of the spaCy model for sentence splitting.
        """
        self.max_tokens = max_tokens
        self.min_overlap = min_overlap
        self.model_name = model_name
        self.spacy_model_name = spacy_model
        
        # Initialize tokenizer (for token counting)
        self._setup_tokenizer()
        
        # Initialize sentence splitter
        self._setup_sentence_splitter()
        
        # Initialize section hierarchy tracker
        self.section_hierarchy = SectionHierarchy()
        
        # Token count cache to avoid repeated calculations
        self._token_count_cache = {}
        
        logger.info(
            f"Initialized TextChunker with max_tokens={max_tokens}, "
            f"min_overlap={min_overlap}, model={model_name}"
        )

    def _setup_tokenizer(self):
        """Set up the tokenizer based on the model."""
        if tiktoken_available:
            try:
                openai_models = {
                    "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k",
                    "gpt-4o", "gpt-4-turbo"
                }
                
                # For OpenAI models, use the specific encoding
                if any(self.model_name.startswith(m) for m in openai_models):
                    self.encoding = tiktoken.encoding_for_model(self.model_name)
                else:
                    # For other models, use cl100k_base as default
                    self.encoding = tiktoken.get_encoding("cl100k_base")
                    
                logger.debug(f"Using tiktoken with {self.model_name} encoding")
            except Exception as e:
                logger.warning(f"Error initializing tiktoken: {e}. Using fallback counting.")
                self.encoding = None
        else:
            logger.warning("tiktoken not available. Using fallback counting.")
            self.encoding = None

    def _setup_sentence_splitter(self):
        """Set up the sentence splitter."""
        if spacy_available:
            try:
                self.nlp = spacy.load(self.spacy_model_name)
                logger.debug(f"Using spaCy model {self.spacy_model_name} for sentence splitting")
            except OSError:
                logger.warning(f"spaCy model '{self.spacy_model_name}' not found. Using regex fallback.")
                self.nlp = None
            except Exception as e:
                logger.warning(f"Error loading spaCy model: {e}. Using regex fallback.")
                self.nlp = None
        else:
            logger.warning("spaCy not available. Using regex fallback for sentence splitting.")
            self.nlp = None

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.
        Uses memoization to avoid recounting the same text.
        
        Args:
            text: The text to count tokens for.
            
        Returns:
            The number of tokens in the text.
        """
        # Check cache first before encoding
        if text in self._token_count_cache:
            return self._token_count_cache[text]
            
        # Count tokens using tiktoken or fallback
        token_count = count_tokens_with_tiktoken(text, self.model_name)
        
        # Cache the result to avoid future encoding
        self._token_count_cache[text] = token_count
        return token_count

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: The text to split.
            
        Returns:
            A list of sentences.
        """
        if self.nlp is not None:
            # Use spaCy for sentence splitting
            return [sent.text.strip() for sent in self.nlp(text).sents if sent.text.strip()]
        else:
            # Fallback to regex-based sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str, repo_link: str, file_path: str) -> List[Dict]:
        """
        Main method to chunk text while preserving section structure.
        
        Args:
            text: The text to chunk.
            repo_link: Link to the repository.
            file_path: Path to the file within the repository.
            
        Returns:
            A list of chunk dictionaries with metadata.
        """
        logger.info(f"Starting chunk_text for file: {file_path}")
        
        # First, split the text into sections
        sections = self._split_by_sections(text)
        extracted_data: List[Dict] = []

        if sections:
            logger.info(f"Found {len(sections)} sections")
            for idx, (sec_num, sec_title, span) in enumerate(sections):
                start, end = span
                section_content = text[start:end].strip()
                self.section_hierarchy.update(sec_num, sec_title, section_content)

                extracted_data.extend(
                    self._chunk_section(
                        text,
                        sec_num,
                        sec_title,
                        section_content,
                        span,
                        repo_link,
                        file_path,
                    )
                )
        else:
            logger.warning("No sections found, using fallback chunking")
            extracted_data.extend(self._fallback_chunking(text, repo_link, file_path))

        logger.info(f"Generated {len(extracted_data)} chunks")
        return extracted_data

    def _split_by_sections(self, text: str) -> List[Tuple[str, str, Tuple[int, int]]]:
        """
        Split text into sections based on markdown headers.
        
        This identifies section markers like:
          **4. General Design Requirements**
          **4 General Plant Design**
          **4.1 Plant Design Basis**
        
        Args:
            text: The text to split into sections.
            
        Returns:
            A list of tuples: (section_number, section_title, (start, end))
        """
        logger.info("Splitting text into sections")
        
        # Define regex pattern for section headers
        section_pattern = re.compile(
            r"^"  # start of line
            r"\*\*"  # **
            r"(?P<number>\d+(?:\.\d+)*\.?)"  # 4 or 4. or 4.1 or 4.1.1.
            r"\s+"  # at least one space
            r"(?P<title>[A-Z0-9][^\n*]+?)"  # title (no newlines allowed)
            r"\s*\*\*"  # optional space, then closing **
            r"[ \t]*$",  # only whitespace until EOL
            re.MULTILINE,
        )

        # Find all section headers
        sections: List[Tuple[str, str, Tuple[int, int]]] = []
        for m in section_pattern.finditer(text):
            raw = m.group("number")
            num = raw.rstrip(".")  # normalize "4." → "4"
            title = m.group("title").strip()
            start, hdr_end = m.start(), m.end()

            # Find the start of the next section (or end of text)
            nxt = section_pattern.search(text, hdr_end)
            end = nxt.start() if nxt else len(text)

            sections.append((num, title, (start, end)))

        # If no sections found, use the entire document as one section
        if not sections:
            sections.append(("", text, (0, len(text))))
        # If there's text after the last section, add it as an untitled section
        elif sections[-1][2][1] < len(text) and text[sections[-1][2][1]:].strip():
            last_end = sections[-1][2][1]
            sections.append(("", text[last_end:].strip(), (last_end, len(text))))

        logger.info(f"Split text into {len(sections)} sections")
        return sections
    
    def _create_chunk_dict(
        self,
        chunk_content: str,
        token_count: int,
        section_title: str,
        start_line: int,
        file_path: str,
        repo_link: str,
    ) -> Dict[str, Any]:
        """
        Helper method to create a chunk dictionary with metadata.
        
        Args:
            chunk_content: The content of the chunk.
            token_count: The token count of the chunk content.
            section_title: The section title.
            start_line: The starting line number.
            file_path: Path to the file within the repository.
            repo_link: Link to the repository.
            
        Returns:
            A dictionary with chunk metadata.
        """
        # Calculate end line based on newlines in content
        end_line = start_line + chunk_content.count("\n")
        
        # Create section hash once
        section_hash = hash_string(chunk_content)
        
        # Cache the section title token count
        if section_title not in self._token_count_cache:
            self.count_tokens(section_title)  # This will cache the result
        
        # Create and return the chunk dictionary
        return {
            "file_path": file_path,
            "repo_link": repo_link,
            "extraction_date": datetime.datetime.now().isoformat(),
            "code_line_span": (start_line, end_line),
            "description_line_span": (start_line, end_line),
            "code": chunk_content,
            "code_type": "text",
            "description": section_title,
            "code_token_count": token_count,
            "description_token_count": self._token_count_cache.get(section_title, self.count_tokens(section_title)),
            "embedding_code": None,
            "embedding_description": None,
            "code_metadata": {},
            "section_id": section_hash,
            "section_path": self.section_hierarchy.get_titles(),
            "section_hash_path": self.section_hierarchy.get_hashes(),
        }
    
    def _chunk_section(
        self,
        text: str,
        section_number: str,
        section_title: str,
        section_content: str,
        span: Tuple[int, int],
        repo_link: str,
        file_path: str,
    ) -> List[Dict]:
        """
        Chunk a single section into smaller pieces based on token limit.
        
        Args:
            text: The full text.
            section_number: The section number string.
            section_title: The section title string.
            section_content: The content of this section.
            span: The (start, end) character offsets in the original text.
            repo_link: Link to the repository.
            file_path: Path to the file within the repository.
            
        Returns:
            A list of chunk dictionaries with metadata.
        """
        logger.info(f"Chunking section: {section_title!r}")
        sentences = self.split_into_sentences(section_content)
        chunks = []
        current_chunk = ""
        token_count = 0
        start_line = span[0] + 1  # 1-indexed line numbering

        for sent in sentences:
            sent_tokens = self.count_tokens(sent)
            
            # If this single sentence is already larger than max_tokens, split it further
            if sent_tokens > self.max_tokens:
                # If we have accumulated content, flush it first
                if current_chunk:
                    # Use the helper method for chunk creation
                    chunk_dict = self._create_chunk_dict(
                        chunk_content=current_chunk,
                        token_count=token_count,
                        section_title=section_title,
                        start_line=start_line,
                        file_path=file_path,
                        repo_link=repo_link
                    )
                    chunks.append(chunk_dict)
                    
                    # Update start line for next chunk
                    start_line = chunk_dict["code_line_span"][1] + 1
                    current_chunk = ""
                    token_count = 0
                
                # Handle the oversized sentence by splitting it into words
                words = sent.split(' ')
                current_sentence_chunk = ""
                current_sentence_token_count = 0
                
                for word in words:
                    word_tokens = self.count_tokens(word + " ")
                    if current_sentence_token_count + word_tokens > self.max_tokens and current_sentence_chunk:
                        # Use the helper method for chunk creation
                        chunk_dict = self._create_chunk_dict(
                            chunk_content=current_sentence_chunk,
                            token_count=current_sentence_token_count,
                            section_title=section_title,
                            start_line=start_line,
                            file_path=file_path,
                            repo_link=repo_link
                        )
                        chunks.append(chunk_dict)
                        
                        # Update start line for next chunk
                        start_line = chunk_dict["code_line_span"][1] + 1
                        current_sentence_chunk = ""
                        current_sentence_token_count = 0
                    
                    current_sentence_chunk += word + " "
                    current_sentence_token_count += word_tokens
                
                # Add any remaining words in the final sentence chunk
                if current_sentence_chunk:
                    chunk_dict = self._create_chunk_dict(
                        chunk_content=current_sentence_chunk,
                        token_count=current_sentence_token_count,
                        section_title=section_title,
                        start_line=start_line,
                        file_path=file_path,
                        repo_link=repo_link
                    )
                    chunks.append(chunk_dict)
                    start_line = chunk_dict["code_line_span"][1] + 1
                
            # Normal case - if adding this sentence would exceed tokens, flush the current chunk
            elif token_count + sent_tokens > self.max_tokens and current_chunk:
                chunk_dict = self._create_chunk_dict(
                    chunk_content=current_chunk,
                    token_count=token_count,
                    section_title=section_title,
                    start_line=start_line,
                    file_path=file_path,
                    repo_link=repo_link
                )
                chunks.append(chunk_dict)
                start_line = chunk_dict["code_line_span"][1] + 1
                
                # Add the current sentence to the start of a new chunk
                current_chunk = sent + "\n"
                token_count = sent_tokens
            else:
                # Add the sentence to the current chunk
                current_chunk += sent + "\n"
                token_count += sent_tokens

        # Flush any remaining content
        if current_chunk:
            # Use the helper method for the final chunk
            chunk_dict = self._create_chunk_dict(
                chunk_content=current_chunk,
                token_count=token_count,
                section_title=section_title,
                start_line=start_line,
                file_path=file_path,
                repo_link=repo_link
            )
            chunks.append(chunk_dict)

        logger.info(f"Section {section_title!r} chunked into {len(chunks)} pieces")
        return chunks
    
    def _fallback_chunking(
        self, text: str, repo_link: str, file_path: str
    ) -> List[Dict]:
        """
        Handle text without identifiable sections using fallback chunking.
        
        Args:
            text: The text to chunk.
            repo_link: Link to the repository.
            file_path: Path to the file within the repository.
            
        Returns:
            A list with a single chunk dictionary.
        """
        logger.info("Performing fallback chunking")
        
        code_token_count = self.count_tokens(text)
        section_hash = hash_string(text)
        code_start_line = 1
        code_end_line = text.count("\n") + 1

        data = {
            "file_path": file_path,
            "repo_link": repo_link,
            "extraction_date": datetime.datetime.now().isoformat(),
            "code_line_span": (code_start_line, code_end_line),
            "description_line_span": (1, 1),
            "code": text,
            "code_type": "text",
            "description": "",
            "code_token_count": code_token_count,
            "description_token_count": 0,
            "embedding_code": None,
            "embedding_description": None,
            "code_metadata": {},
            "section_id": section_hash,
            "section_path": self.section_hierarchy.get_titles(),
            "section_hash_path": self.section_hierarchy.get_hashes(),
        }
        logger.info(f"Created fallback chunk with section_id={section_hash}")
        return [data]

if __name__ == "__main__":
    """Validate text chunking with real data"""
    import sys
    from rich.console import Console
    from rich.table import Table
    import argparse
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    console = Console()
    
    # Sample text for validation
    sample_text = """
**1. Introduction**

This is a sample document to demonstrate the text chunking functionality.
It has multiple sections with different levels.

**1.1 Purpose**

The purpose of this document is to verify that the text chunker correctly:
1. Identifies section headers
2. Maintains section hierarchy
3. Chunks text within token limits
4. Preserves metadata across chunks

**2. Features**

The text chunker has several important features.

**2.1 Section Detection**

The chunker can detect section headers like this one and build a hierarchy.

**2.2 Token Counting**

The chunker uses tiktoken to accurately count tokens for various models.

**3. Conclusion**

This sample document should be sufficient to verify the basic functionality.
"""
    
    # Test 1: Token counting
    total_tests += 1
    try:
        # Create a chunker
        chunker = TextChunker(max_tokens=500, min_overlap=100, model_name="gpt-4")
        
        # Test token counting
        test_strings = [
            "This is a short sentence.",
            "This is a longer sentence with more words and should have more tokens.",
            "Technical terms like 'tokenization' and 'chunking' might be counted as single tokens.",
        ]
        
        # Verify token counts are reasonable
        for text in test_strings:
            token_count = chunker.count_tokens(text)
            char_count = len(text)
            ratio = char_count / token_count if token_count > 0 else 0
            
            # Check if the ratio is within a reasonable range
            # Typically 4-7 characters per token for English text
            if not (3 <= ratio <= 8):
                all_validation_failures.append(f"Token counting test: Unusual ratio {ratio} for '{text}'")
    except Exception as e:
        all_validation_failures.append(f"Token counting test failed: {str(e)}")
    
    # Test 2: Section detection
    total_tests += 1
    try:
        # Extract sections from sample text
        sections = chunker._split_by_sections(sample_text)
        
        # Verify sections
        if len(sections) != 7:  # We expect 7 sections
            all_validation_failures.append(f"Section detection test: Expected 7 sections, got {len(sections)}")
        
        # Check section numbers
        expected_section_numbers = ["1", "1.1", "2", "2.1", "2.2", "3", ""]
        actual_section_numbers = [s[0] for s in sections]
        
        if actual_section_numbers != expected_section_numbers:
            all_validation_failures.append(f"Section detection test: Expected section numbers {expected_section_numbers}, got {actual_section_numbers}")
    except Exception as e:
        all_validation_failures.append(f"Section detection test failed: {str(e)}")
    
    # Test 3: Section hierarchy
    total_tests += 1
    try:
        # Create a new section hierarchy
        hierarchy = SectionHierarchy()
        
        # Update with sections
        hierarchy.update("1", "Introduction", "Content 1")
        hierarchy.update("1.1", "Purpose", "Content 1.1")
        hierarchy.update("2", "Features", "Content 2")
        
        # Verify titles
        expected_titles = ["1 Introduction", "1.1 Purpose"]
        actual_titles = hierarchy.get_titles()
        
        if len(actual_titles) != 2:
            all_validation_failures.append(f"Section hierarchy test: Expected 2 titles, got {len(actual_titles)}")
        
        # Check that 1.1 is a child of 1
        if "1 Introduction" not in actual_titles or "1.1 Purpose" not in actual_titles:
            all_validation_failures.append(f"Section hierarchy test: Expected titles {expected_titles}, got {actual_titles}")
    except Exception as e:
        all_validation_failures.append(f"Section hierarchy test failed: {str(e)}")
    
    # Test 4: Chunk generation
    total_tests += 1
    try:
        # Create a chunker
        chunker = TextChunker(max_tokens=200, min_overlap=50, model_name="gpt-4")
        
        # Generate chunks
        chunks = chunker.chunk_text(
            sample_text,
            "https://example.com/repo",
            "example/file.md"
        )
        
        # Verify chunks
        if len(chunks) < 3:  # We expect at least 3 chunks
            all_validation_failures.append(f"Chunk generation test: Expected at least 3 chunks, got {len(chunks)}")
        
        # Check chunk structure
        for i, chunk in enumerate(chunks):
            # Check required fields
            required_fields = ["file_path", "code", "code_token_count", "section_path", "section_id"]
            missing_fields = [field for field in required_fields if field not in chunk]
            
            if missing_fields:
                all_validation_failures.append(f"Chunk {i} missing required fields: {missing_fields}")
            
            # Check token count is under the limit
            if chunk["code_token_count"] > chunker.max_tokens:
                all_validation_failures.append(f"Chunk {i} exceeds token limit: {chunk['code_token_count']} > {chunker.max_tokens}")
    except Exception as e:
        all_validation_failures.append(f"Chunk generation test failed: {str(e)}")
    
    # Test 5: Hash string generation
    total_tests += 1
    try:
        # Test hash_string function
        test_strings = [
            "Test string 1",
            "Test string 2",
            "Completely different string"
        ]
        
        # Generate hashes
        hashes = [hash_string(s) for s in test_strings]
        
        # Verify hashes are different
        if len(set(hashes)) != len(test_strings):
            all_validation_failures.append("Hash string test: Duplicate hashes generated for different strings")
        
        # Verify hash length
        if any(len(h) != 64 for h in hashes):
            all_validation_failures.append("Hash string test: Invalid hash length (should be 64 characters)")
        
        # Verify hash stability
        if hash_string(test_strings[0]) != hashes[0]:
            all_validation_failures.append("Hash string test: Hash function is not stable")
    except Exception as e:
        all_validation_failures.append(f"Hash string test failed: {str(e)}")
    
    # Test 6: Sentence splitting
    total_tests += 1
    try:
        # Test sentence splitting
        test_text = "This is the first sentence. This is the second sentence! And this is the third?"
        
        # Split sentences
        sentences = chunker.split_into_sentences(test_text)
        
        # Verify sentence count
        if len(sentences) != 3:
            all_validation_failures.append(f"Sentence splitting test: Expected 3 sentences, got {len(sentences)}")
        
        # Verify sentence content
        expected_first = "This is the first sentence."
        if sentences[0] != expected_first:
            all_validation_failures.append(f"Sentence splitting test: Expected first sentence '{expected_first}', got '{sentences[0]}'")
    except Exception as e:
        all_validation_failures.append(f"Sentence splitting test failed: {str(e)}")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Text chunking functions are validated and ready for use")
        sys.exit(0)  # Exit with success code