```markdown
# Task: Debug and Implement PDF Extraction Service with Optional HITL Workflow

This task involves debugging and verifying the complete pipeline for extracting structured data (tables, headings, paragraphs) from PDFs, with an optional Human-in-the-Loop (HITL) workflow for validating tables using Label Studio. The pipeline uses a **FastAPI server** (`api.py`) for Dockerized deployment, a **Typer CLI** (`cli.py`) for local deployment, and `pdf_converter.py` for core extraction logic, deployed via Docker Compose. The goal is to ensure all components work seamlessly, producing a correct `structured.json` output, with Server-Sent Events (SSE) for Multi-Cloud Platform (MCP) compatibility.

## I. Core Components

* [ ] `Dockerfile`:
  * [ ] Verify the image builds successfully with all dependencies (e.g., Poppler, Ghostscript, Tesseract).
  * [ ] Confirm `api.py`, `cli.py`, `pdf_converter.py`, `config.py`, `utils.py`, `table_extraction.py`, `marker_processor.py`, `qwen_processor.py` are copied correctly.
  * [ ] Ensure `uvicorn` runs the FastAPI server on port 8000.
  * [ ] Debug issues with missing system dependencies (e.g., `libtesseract-dev`).

* [ ] `docker-compose.yml`:
  * [ ] Confirm both services (`labelstudio`, `fastapi`) start without errors.
  * [ ] Verify shared volumes (`uploads`, `output`, `corrections`, `label-studio-data`) are mounted correctly.
  * [ ] Ensure networking allows Label Studio to access FastAPI if integrated (`http://fastapi:8000`).
  * [ ] Debug connectivity issues (e.g., port 8000 conflicts, volume permissions).

* [ ] `requirements.txt`:
  * [ ] Validate all Python dependencies install without conflicts (e.g., `fastapi`, `sse-starlette`, `marker-pdf`, `transformers`, `camelot-py`).
  * [ ] Check for version mismatches causing runtime errors (e.g., `torch` vs. `transformers`).
  * [ ] Add missing dependencies if errors occur (e.g., `python-multipart`, `typer`).

* [ ] `api.py`:
  * [ ] Implement and verify FastAPI endpoints:
    * [ ] `POST /convert`: Uploads PDF, runs extraction, returns `structured.json`.
    * [ ] `POST /stream/convert`: Streams extraction progress via SSE.
    * [ ] `GET /status`: Returns server status.
  * [ ] Debug endpoint failures (e.g., file not found, JSON parsing errors, SSE stream interruptions).
  * [ ] Ensure `usage_function` block runs independently, processing a test PDF.
  * [ ] (Optional) If Label Studio integration is needed, add and verify:
    * [ ] `POST /upload`: Generates Label Studio tasks.
    * [ ] `GET /pdf/<pdf_id>/page/<page>`: Serves PDF page images.
    * [ ] `POST /save/<pdf_id>`: Saves Label Studio annotations.
    * [ ] `POST /reextract/<pdf_id>`: Re-extracts with corrections.

* [ ] `cli.py`:
  * [ ] Verify the Typer CLI command:
    * [ ] `convert`: Converts PDF to `structured.json`, mirroring `/convert`.
  * [ ] Debug CLI failures (e.g., invalid PDF path, output directory issues).
  * [ ] Ensure `usage_function` block runs independently, producing `structured.json`.

* [ ] `pdf_converter.py`:
  * [ ] Verify core functions:
    * [ ] Table extraction with Camelot, Marker, and Qwen-VL-2B.
    * [ ] Text normalization using `cleantext`.
    * [ ] Integration of corrections (if Label Studio is used).
  * [ ] Debug extraction issues (e.g., Camelot failing on complex layouts, Qwen-VL-2B memory errors).
  * [ ] Ensure `if __name__ == "__main__":` block processes a test PDF, producing `structured.json`.
  * [ ] Confirm compatibility with both `api.py` and `cli.py`.

* [ ] `config.py`:
  * [ ] Verify `DEFAULT_OUTPUT_DIR` (`output`) and `DEFAULT_CORRECTIONS_DIR` (`corrections`) match volume mounts.
  * [ ] Debug issues with environment variable overrides (e.g., `OUTPUT_DIR`, `CORRECTIONS_DIR`).

* [ ] `utils.py`, `table_extraction.py`, `marker_processor.py`, `qwen_processor.py`:
  * [ ] Verify utility functions support `pdf_converter.py` (e.g., table merging, text processing).
  * [ ] Debug specific issues (e.g., fuzzy matching in `utils.py`, OCR errors in `qwen_processor.py`).
  * [ ] Ensure modularity and error handling.

* [ ] `labeling_config.xml` (Optional):
  * [ ] Verify Label Studio loads the interface correctly if integration is enabled.
  * [ ] Confirm elements work:
    * [ ] PDF page rendering with bounding boxes.
    * [ ] Table editing (headers, body).
    * [ ] Validation choices (Approve, Reject, Edit, Add Table).
    * [ ] Merge instruction input.
    * [ ] Parameter inputs (e.g., `flavor`, `line_scale`).
  * [ ] Debug rendering issues (e.g., PDF pages not loading, bounding boxes misaligned).

* [ ] `README.md`:
  * [ ] Validate setup instructions work (e.g., `docker compose up --build`, `python cli.py convert`).
  * [ ] Confirm API and CLI usage examples (`curl`, CLI commands) are correct.
  * [ ] Ensure architecture diagram and project structure reflect the implementation.
  * [ ] Update if debugging reveals new dependencies or steps.

## II. Debugging and Verification Pipeline

### 1. Setup and Deployment
* [ ] **Docker Environment**:
  * [ ] Run `docker compose up --build` and check for build errors in `fastapi` image.
  * [ ] Verify services are accessible:
    * [ ] FastAPI: `http://localhost:8000` (Swagger UI: `http://localhost:8000/docs`)
    * [ ] Label Studio: `http://localhost:8080` (if enabled)
  * [ ] Debug volume mounting issues (e.g., `corrections/` not writable).
  * [ ] Ensure `label-studio-data/` persists Label Studio project data.
* [ ] **Dependency Check**:
  * [ ] Confirm system dependencies (e.g., Poppler, Tesseract) are installed in `fastapi` container.
  * [ ] Test Python dependencies: `docker exec fastapi pip install -r requirements.txt`.
  * [ ] Debug missing dependencies (e.g., `libsm6` for OpenCV, `sse-starlette`).

### 2. Standalone Script Verification
* [ ] **pdf_converter.py**:
  * [ ] Run `if __name__ == "__main__":` with a test PDF (`sample.pdf`):
    ```bash
    python pdf_converter.py
    ```
  * [ ] Verify `output/sample_structured.json` contains expected tables, headings, and metadata.
  * [ ] Debug issues:
    * [ ] Camelot: Check for layout errors (e.g., incorrect table boundaries).
    * [ ] Marker: Ensure Markdown parsing extracts tables correctly.
    * [ ] Qwen-VL-2B: Test on a scanned PDF, debug memory or model loading errors.
* [ ] **api.py**:
  * [ ] Run `usage_function`:
    ```bash
    python api.py
    ```
  * [ ] Test endpoints with `curl`:
    * [ ] Convert: `curl -X POST -F "file=@sample.pdf" -F "repo_link=https://github.com/example/repo" http://localhost:8000/convert`
    * [ ] Stream: `curl http://localhost:8000/stream/convert -F "file=@sample.pdf" -F "repo_link=https://github.com/example/repo"`
    * [ ] Status: `curl http://localhost:8000/status`
  * [ ] Debug failures (e.g., 400 for invalid PDFs, SSE connection drops).
* [ ] **cli.py**:
  * [ ] Run `usage_function` or command:
    ```bash
    python cli.py convert sample.pdf --repo-link https://github.com/example/repo
    ```
  * [ ] Verify `output/sample_structured.json` matches API output.
  * [ ] Debug CLI issues (e.g., path resolution, parameter parsing).

### 3. Label Studio Integration (Optional)
* [ ] **Configuration**:
  * [ ] Upload `labeling_config.xml` to Label Studio via the UI.
  * [ ] Debug interface issues (e.g., missing table editing grid).
* [ ] **Task Import**:
  * [ ] If `api.py` includes `/upload`, import `corrections/<pdf_id>_tasks.json` into Label Studio.
  * [ ] Verify tasks load with correct PDF pages and table data.
  * [ ] Debug import failures (e.g., invalid JSON).
* [ ] **Annotation**:
  * [ ] Test annotation workflow:
    * [ ] Approve a correct table.
    * [ ] Edit a table’s headers/body.
    * [ ] Reject an incorrect table.
    * [ ] Add a new table manually.
    * [ ] Mark a multi-page merge (e.g., “Merge with table_2”).
    * [ ] Set parameters (e.g., `flavor: stream`, `line_scale: 50`).
  * [ ] Debug annotation issues (e.g., bounding boxes not rendering).
* [ ] **Export**:
  * [ ] Export annotations as JSON and verify structure.
  * [ ] Debug export issues (e.g., missing fields).

### 4. End-to-End Workflow
* [ ] **Test Case 1: Text-Based PDF**:
  * [ ] Upload a PDF via `/convert` or CLI.
  * [ ] Verify `output/<pdf_id>_structured.json` contains tables, headings, and metadata.
  * [ ] Debug extraction issues (e.g., missing tables, incorrect metadata).
* [ ] **Test Case 2: Scanned PDF**:
  * [ ] Upload a scanned PDF.
  * [ ] Verify Qwen-VL-2B extracts content correctly.
  * [ ] Debug OCR issues (e.g., low accuracy, model crashes).
* [ ] **Test Case 3: SSE Streaming**:
  * [ ] Use `/stream/convert` to process a PDF.
  * [ ] Verify SSE events (start, progress, complete) are received.
  * [ ] Debug streaming issues (e.g., connection drops, missing events).
* [ ] **Test Case 4: Label Studio (if enabled)**:
  * [ ] Upload a PDF via `/upload`, import tasks, edit a table, save corrections, and re-extract.
  * [ ] Confirm `structured.json` reflects human edits with `source: HUMAN`.
  * [ ] Debug correction integration (e.g., merge instructions ignored).

### 5. Correction Integration (Optional)
* [ ] Verify `corrections/<pdf_id>_corrections.json` is correctly formatted:
  ```json
  {
    "pdf_path": "uploads/sample.pdf",
    "tables": [
      {
        "page": 1,
        "status": "approved",
        "header": ["Column1", "Column2"],
        "body": [["Cell1", "Cell2"]],
        "source": "CAMELOT"
      }
    ],
    "reextract_pages": []
  }
  ```
* [ ] Confirm `pdf_converter.py` applies corrections (if implemented):
  * [ ] Approved tables are unchanged.
  * [ ] Edited tables reflect human changes.
  * [ ] Rejected tables are excluded.
  * [ ] Added tables are included with `source: HUMAN`.
* [ ] Debug correction parsing errors (e.g., invalid JSON).

## III. Debugging Steps

* [ ] **Logs**:
  * [ ] Enable `loguru` debug logging in `api.py`, `cli.py`, and `pdf_converter.py`.
  * [ ] Check FastAPI logs for endpoint errors (e.g., 500 Internal Server Error).
  * [ ] Check Label Studio logs for task import/export issues (if enabled).
  * [ ] Debug extraction failures using `pdf_converter.py` logs (e.g., Camelot exceptions).
* [ ] **Dependency Issues**:
  * [ ] If `fastapi` container crashes, inspect logs for missing libraries (e.g., `libxrender-dev`).
  * [ ] Rebuild with updated `requirements.txt` if Python dependencies fail (e.g., add `numpy==1.26.2`).
* [ ] **Network Issues**:
  * [ ] If Label Studio cannot access `http://fastapi:8000`, verify Docker network (`pdf-extraction-network`).
  * [ ] Test connectivity: `docker exec labelstudio curl http://fastapi:8000`.
* [ ] **Performance**:
  * [ ] If Qwen-VL-2B is slow, reduce `max_new_tokens` or use CPU.
  * [ ] Optimize Camelot by limiting pages in `pdf_converter.py` for large PDFs.
* [ ] **Data Validation**:
  * [ ] Validate `output/<pdf_id>_structured.json` against expected schema.
  * [ ] Debug JSON parsing errors in `api.py` or `cli.py`.

## IV. Infrastructure and Tooling

* [ ] **Docker Verification**:
  * [ ] Confirm `Dockerfile` installs system dependencies (e.g., `tesseract-ocr`, `poppler-utils`).
  * [ ] Test `docker-compose.yml`:
    ```bash
    docker compose down && docker compose up --build
    ```
  * [ ] Debug service startup failures (e.g., port conflicts, volume permissions).
* [ ] **Volume Management**:
  * [ ] Ensure volumes (`uploads`, `output`, `corrections`) are writable.
  * [ ] Debug permission issues: `docker exec fastapi chmod -R 777 /app/corrections`.

## V. Code Quality and Standards

* [ ] **Docstrings**:
  * [ ] Verify `api.py`, `cli.py`, `pdf_converter.py`, and supporting scripts have module-level docstrings with purpose, third-party links, and sample input/output.
  * [ ] Add missing docstrings for new functions (e.g., SSE streaming in `api.py`).
* [ ] **Standalone Verification**:
  * [ ] Ensure `pdf_converter.py`, `api.py`, and `cli.py` have `if __name__ == "__main__":` blocks:
    ```bash
    python pdf_converter.py
    python api.py
    python cli.py convert sample.pdf
    ```
* [ ] **Logging**:
  * [ ] Confirm `loguru` logs critical events (e.g., extraction start, errors, SSE events).
  * [ ] Add debug logs for CLI commands and SSE streaming if missing.
* [ ] **README Alignment**:
  * [ ] Update `README.md` with new setup steps or dependencies discovered during debugging.
  * [ ] Validate all `curl` and CLI examples.

## VI. Testing Strategy

* [ ] **Pre-Pytest Verification**:
  * [ ] Ensure `api.py`, `cli.py`, and `pdf_converter.py` usage functions work before pytest.
  * [ ] Manually test each endpoint and CLI command with test PDFs (text-based, scanned).
* [ ] **Manual Testing**:
  * [ ] Run end-to-end workflow for each test case (text-based, scanned, streaming).
  * [ ] Verify `structured.json` for correctness (e.g., tables, metadata).
* [ ] **Pytest (Post-Verification)**:
  * [ ] Create `tests/unit/test_pdf_converter.py` for extraction functions.
  * [ ] Create `tests/integration/test_api.py` for FastAPI endpoints.
  * [ ] Create `tests/integration/test_cli.py` for CLI commands.
  * [ ] Run tests:
    ```bash
    docker exec fastapi pip install pytest
    docker exec fastapi pytest /app/tests
    ```

## VII. Expected Output

* [ ] **output/<pdf_id>_structured.json**:
  ```json
  [
    {
      "type": "table",
      "header": ["Column1", "Column2"],
      "body": [["Cell1", "Cell2"], ["Cell3", "Cell4"]],
      "page_range": [1, 1],
      "token_count": 12,
      "file_path": "uploads/sample.pdf",
      "repo_link": "https://github.com/example/repo",
      "extraction_date": "2025-04-19T12:00:00.000000",
      "source": "CAMELOT",
      "needs_review": false
    },
    {
      "type": "heading",
      "level": 1,
      "text": "Introduction",
      "token_count": 1,
      "file_path": "uploads/sample.pdf",
      "repo_link": "https://github.com/example/repo",
      "extraction_date": "2025-04-19T12:00:00.000000",
      "source": "MARKER_JSON",
      "needs_review": false
    }
  ]
  ```

## VIII. Debugging Checklist

* [ ] **Build Failures**:
  * [ ] Check `Dockerfile` for missing system dependencies (e.g., `libjpeg-dev`).
  * [ ] Update `requirements.txt` for Python dependency conflicts.
* [ ] **Service Failures**:
  * [ ] Inspect `docker compose logs fastapi` and `docker compose logs labelstudio`.
  * [ ] Fix port conflicts (e.g., change 8000 in `docker-compose.yml`).
* [ ] **Extraction Issues**:
  * [ ] Test Camelot with different settings in `table_extraction.py`.
  * [ ] Reduce Qwen-VL-2B memory usage in `qwen_processor.py`.
  * [ ] Verify Marker parses tables in `marker_processor.py`.
* [ ] **SSE Issues**:
  * [ ] Ensure `/stream/convert` sends all events (start, progress, complete).
  * [ ] Debug client connection drops or event parsing errors.
* [ ] **Label Studio Issues (if enabled)**:
  * [ ] Ensure `pdf_page_url` resolves to `http://fastapi:8000/pdf/<pdf_id>/page/<page>`.
  * [ ] Fix annotation export format mismatches.

## IX. Agentic Workflow

* **Planner**: Break down debugging tasks (e.g., verify Docker, test endpoints, debug SSE).
* **Coder**: Fix code issues (e.g., add error handling in `api.py`, optimize Qwen-VL-2B).
* **Debugger**: Analyze logs, reproduce failures, and suggest fixes (e.g., dependency updates).
* **Tester**: Run manual tests for each test case, verify outputs.
* **Documenter**: Update `README.md`, `task.md`, and docstrings with debugging insights.

## X. Success Criteria

* [ ] Docker services start without errors.
* [ ] `pdf_converter.py` produces correct `structured.json` for test PDFs.
* [ ] FastAPI endpoints (`/convert`, `/stream/convert`, `/status`) handle PDF processing and streaming.
* [ ] Typer CLI (`convert`) produces identical output to `/convert`.
* [ ] (Optional) Label Studio integration works if implemented (tasks load, annotations saved, corrections applied).
* [ ] End-to-end workflow completes for text-based and scanned PDFs, with SSE streaming verified.
```