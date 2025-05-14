# Task: Debug and Implement PDF Extraction Service with HITL Workflow

This task involves debugging and verifying a semi-automated pipeline for extracting structured data (tables, headings, paragraphs) from PDFs, with a Human-in-the-Loop (HITL) workflow. The HITL workflow is used for validating tables, marking multi-page merges, and tweaking extraction parameters. The pipeline uses a Flask API (`app.py`) for orchestration, `pdf_to_json_converter.py` for extraction, and LabelStudio for human validation, deployed via Docker Compose. The goal is to ensure all components work seamlessly, producing a correct `structured.json` output, while allowing for human oversight and ground truth generation. PDF extraction is inherently messy, so the pipeline is designed with multiple fallbacks and opportunities for human intervention.

**I. Core Components:**

* [X] `Dockerfile`:
    * [X] Verify the image builds successfully with all dependencies (e.g., Poppler, Ghostscript, Tesseract).
    * [X] Confirm `app.py` and `pdf_to_json_converter.py` are copied correctly.
    * [X] Ensure `gunicorn` runs the Flask API on port 5000.
    * [X] Debug issues with missing system dependencies (e.g., `libtesseract-dev`).

* [ ] `docker-compose.yml`:
    * [ ] Confirm both services (`labelstudio`, `flask-api`) start without errors.
    * [ ] Verify shared volumes (`uploads`, `output`, `corrections`, `label-studio-data`) are mounted correctly.
    * [ ] Ensure networking allows LabelStudio to access Flask API (`http://flask-api:5000`).
    * [ ] Debug connectivity issues between services (e.g., LabelStudio failing to fetch PDF pages).

* [ ] `requirements.txt`:
    * [ ] (Deprecated) This file will be replaced by pyproject.toml.
    * [ ] Validate all Python dependencies install without conflicts (e.g., `marker-pdf`, `transformers`, `camelot-py`).
    * [ ] Check for version mismatches causing runtime errors (e.g., `torch` vs. `transformers`).
    * [ ] Add missing dependencies if errors occur (e.g., `gunicorn` version).
    * [ ] Use UV and update `pyproject.toml`
    *    ```bash
           pip install uv
           uv pip install -e .
           uv pip install -r requirements.txt # if reqirements.txt has been updated
           uv venv # creates a virtual enviroment
           source .venv/bin/activate # enables the new virtual envirnment
           deactivate # get out of the virtual enviroment
           ```

* [ ] `app.py`:
    * [ ] Implement and verify the Flask API endpoints:
        * [ ] `POST /upload`: Uploads PDF, runs initial extraction, generates LabelStudio tasks.
            *   **Original Issue:** The `POST /upload` endpoint was returning a 500 Internal Server Error due to a `FileNotFoundError`. The Flask API was trying to access the PDF from the root directory instead of the `uploads` directory.
            *   **Fix:** Modified the `pdf_path` variable in `app.py` to correctly join the filename with the `UPLOAD_DIR`.
            *   **Verification Steps:**
                1.  Run `curl -X POST -F "file=@input.pdf" http://localhost:5000/upload`
                2.  Verify that the response is 200 OK and contains the expected JSON:
                    ```json
                    {
                        "pdf_id": "input",
                        "tasks_path": "/app/corrections/input_tasks.json"
                    }
                    ```
                *   **Improved Logging:** Added logger in `app.py` to print the actual path being used.
                ```python
                from loguru import logger
                logger.error(f"Error saving file {pdf_path}")
                ```

        * [ ] `GET /pdf/<pdf_id>/page/<page>`: Serves PDF page images.
        * [ ] `POST /save/<pdf_id>`: Converts LabelStudio annotations to corrections JSON.
        * [ ] `POST /reextract/<pdf_id>`: Re-extracts PDF with corrections.
        *   **Test and Validation.**

            1. Start docker and upload input.pdf to the upload endpoint
            2. Check if the endpoint functions without any container errors
            3. Check if the label studio tasks are created inside the appropriate corrections volume
            4. Run test to validate if label studio can access these tasks

    * [ ] Debug endpoint failures (e.g., file not found, JSON parsing errors).
    * [ ] Ensure `usage_example` block runs independently, processing a test PDF.

* [ ] `pdf_to_json_converter.py`:
    * [ ] **IMPORTANT:** Before making any changes to this file, carefully read through the existing code to understand the extraction method priorities and Camelot tuning logic.
    * [ ] Verify core functions:
        * [ ] `extract_tables_with_camelot`: Extracts tables with human-specified parameters.
        * [ ] `merge_camelot_tables` and `merge_multi_page_tables`: Handles automated and human-directed merges.
        * [ ] `process_tables`: Prioritizes human-validated tables and skips rejected ones.
        * [ ] `convert_pdf_to_json`: Integrates Camelot, Marker, Qwen-VL-2B, and corrections.
    * [ ] Debug extraction issues (e.g., Camelot failing on complex layouts, Qwen-VL-2B memory errors).
    * [ ] Ensure `usage_example` block processes a test PDF independently, producing `structured.json`.
    * [ ] Confirm `cleantext` normalization in `get_text_elements_around_table`.

* [ ] `labeling_config.xml`:
    * [ ] Verify LabelStudio loads the interface correctly.
    * [ ] Confirm all elements work:
        * [ ] PDF page rendering with bounding boxes.
        * [ ] Table editing (headers, body).
        * [ ] Validation choices (Approve, Reject, Edit, Add Table).
        * [ ] Merge instruction input.
        * [ ] Parameter inputs (`flavor`, `line_scale`, `edge_tol`).
    * [ ] Debug rendering issues (e.g., PDF pages not loading, bounding boxes misaligned).

* [ ] `README.md`:
    * [ ] Validate setup instructions work (e.g., `docker compose up --build`).
    * [ ] Confirm API usage examples (`curl` commands) are correct.
    * [ ] Ensure architecture diagram and project structure reflect the implementation.
    * [ ] Update if debugging reveals new dependencies or steps.

**II. Debugging and Verification Pipeline:**

1.  **Setup and Deployment (CRITICAL for MVP)**:
    * [ ] **Docker Environment**:
        * [ ] Run `docker compose up --build` and check for build errors in `flask-api` image.
        * [ ] Verify services are accessible:
            * [ ] LabelStudio: `http://localhost:8080`
            * [ ] Flask API: `http://localhost:5000`
                *   **Automated Checkpoint:** Run `docker compose ps` and verify that the `labelstudio` and `flask-api` services have a status of "running" or "healthy".
                ```bash
                docker compose ps
                ```
                *   *Expected Output*:
                    ```
                    NAME                COMMAND                  SERVICE             STATUS              PORTS
                    pdfextractor_flask-api_1   "gunicorn --bind 0.0.0.0:…"   flask-api           running             0.0.0.0:5000->5000/tcp
                    pdfextractor_labelstudio_1   "/entrypoint.sh"         labelstudio         running             0.0.0.0:8080->8080/tcp
                    ```

        * [ ] Debug volume mounting issues (e.g., `corrections/` not writable).
        * [ ] Ensure `label-studio-data/` persists LabelStudio project data.

    * [ ] **Dependency Check**:
        * [ ] Confirm system dependencies (e.g., Poppler, Tesseract) are installed in `flask-api` container.
        * [ ] Test Python dependencies by running `uv pip install .` in a temporary container.
        * [ ] Debug missing dependencies (e.g., `libsm6` for OpenCV).

2.  **Standalone Script Verification and Endpoint Testing (HIGH Priority)**:
    * [ ] **pdf_to_json_converter.py**:
        * [ ] Run `usage_example` with a test PDF (`input.pdf`):
            ```bash
            python pdf_to_json_converter.py
            ```
            *   Verify `output/structured.json` contains expected tables, headings, and metadata.
            *   **Test PDF**: Locate and inspect the `input.pdf`. Verify the expected tables based on that.
        * [ ] Debug issues:
            * [ ] Camelot: Check for layout errors (e.g., incorrect `flavor`).
            * [ ] Marker: Ensure Markdown parsing extracts tables correctly.
            * [ ] Qwen-VL-2B: Test on a scanned PDF, debug memory or model loading errors.
            * [ ] Merging: Confirm automated merges (header similarity) and `needs_review` flags.
        * [ ] Test with a multi-page table PDF, ensuring `page_range` spans correctly.
    * [ ] **app.py**:
        * [ ] Run `usage_example` (if added) or manually test endpoints:
            ```bash
            python app.py
            ```
        * [ ] Test endpoints with `curl`:
            * [ ] Upload: `curl -X POST -F "file=@input.pdf" http://localhost:5000/upload`
                *    **Test:** Upload and check the response for task creation
            * [ ] Page serving: `curl http://localhost:5000/pdf/input/page/1 --output page1.png`
                 *    **Test:** Test all pages and check if the outputted image looks well formated
            * [ ] Save corrections: `curl -X POST -H "Content-Type: application/json" -d @annotations.json http://localhost:5000/save/input`
                 *   **Test:** Check is you can upload the corrections and if an output message for the corrections folder is made
            * [ ] Re-extract: `curl -X POST http://localhost:5000/reextract/input`
                *   **Test:** Check if the re-extracted data looks well formated and structured
        * [ ] Debug endpoint failures (e.g., 404 for pages, JSON validation errors).
        * [ ] Verify `corrections/<pdf_id>_tasks.json` is generated correctly.

3.  **LabelStudio Integration (Medium Priority)**:
    * [ ] **Configuration**:
        * [ ] Upload `labeling_config.xml` to LabelStudio via the UI.
        * [ ] Debug interface issues (e.g., missing table editing grid, non-functional choices).
    * [ ] **Task Import**:
        * [ ] After uploading a PDF, import `corrections/<pdf_id>_tasks.json` into LabelStudio.
        * [ ] Verify tasks load with correct PDF pages and table data.
        * [ ] Debug import failures (e.g., invalid JSON, missing `pdf_page_url`).
    * [ ] **Annotation**:
        * [ ] Test annotation workflow:
            * [ ] Approve a correct table.
            * [ ] Edit a table’s headers/body.
            * [ ] Reject an incorrect table.
            * [ ] Add a new table manually.
            * [ ] Mark a multi-page merge (e.g., “Merge with table_2”).
            * [ ] Set parameters (e.g., `flavor: stream`, `line_scale: 50`).
        * [ ] Debug annotation issues (e.g., bounding boxes not rendering, merge instructions not saving).
    * [ ] **Export**:
        * [ ] Export annotations as JSON and verify structure matches expected format.
        * [ ] Debug export issues (e.g., missing fields, incorrect values).

4.  **End-to-End Workflow (Medium Priority)**:
    * [ ] **Test Case 1: Text-Based PDF with Multi-Page Table**:
        * [ ] Upload a PDF with a table spanning pages 1-2.
        * [ ] Verify initial `structured.json` flags low-confidence tables (`needs_review: true`).
        * [ ] Import tasks into LabelStudio, mark “Merge with table_1” for page 2 table.
        * [ ] Save corrections and re-extract.
        * [ ] Confirm `structured.json` shows a single table with `page_range: [1, 2]`.
        * [ ] Debug merging failures (e.g., incorrect header matching, human instructions ignored).
    * [ ] **Test Case 2: Scanned PDF**:
        * [ ] Upload a scanned PDF.
        * [ ] Verify Qwen-VL-2B extracts tables, marked for review.
        * [ ] Edit a table in LabelStudio, approve, and re-extract.
        * [ ] Confirm `structured.json` includes human-edited table with `source: HUMAN`.
        * [ ] Debug OCR issues (e.g., low accuracy, model crashes).
    * [ ] **Test Case 3: Parameter Tweaking**:
        * [ ] Upload a PDF with poor Camelot extraction (accuracy < 70%).
        * [ ] In LabelStudio, set `flavor: stream`, `line_scale: 50`, `edge_tol: 600`.
        * [ ] Save corrections and re-extract.
        * [ ] Verify improved table accuracy in `structured.json`.
        * [ ] Debug parameter application (e.g., Camelot ignoring `table_areas`).

5.  **Correction Integration (Low Priority)**:
    * [ ] Verify `corrections/<pdf_id>.json` is correctly formatted:
        ```json
        {
            "pdf_path": "uploads/input.pdf",
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
    * [ ] Confirm `pdf_to_json_converter.py` applies corrections:
        * [ ] Approved tables are unchanged.
        * [ ] Edited tables reflect human changes.
        * [ ] Rejected tables are excluded.
        * [ ] Added tables are included with `source: HUMAN`.
        * [ ] Human-directed merges combine tables correctly.
    * [ ] Debug correction parsing errors (e.g., missing fields, invalid merge instructions).

**III. Debugging Steps:**

* [ ] **Logs**:
    * [ ] Enable `loguru` debug logging in `app.py` and `pdf_to_json_converter.py`.
    * [ ] Check Flask logs for endpoint errors (e.g., 500 Internal Server Error).
    * [ ] Check LabelStudio logs for task import/export issues.
    * [ ] Debug extraction failures using `pdf_to_json_converter.py` logs (e.g., Camelot exceptions, Qwen-VL-2B timeouts).
* [ ] **Dependency Issues**:
    * [ ] If `flask-api` container crashes, inspect logs for missing libraries (e.g., `libxrender-dev`).
    * [ ] Rebuild with updated `requirements.txt` or `pyproject.toml` if Python dependencies fail (e.g., add `numpy==1.26.2` explicitly).
* [ ] **Network Issues**:
    * [ ] If LabelStudio cannot fetch PDF pages (`http://flask-api:5000`), verify Docker network (`pdf-extraction-network`).
    * [ ] Test connectivity: `docker exec labelstudio curl http://flask-api:5000`.
* [ ] **Performance**:
    * [ ] If Qwen-VL-2B is slow or crashes, reduce `max_new_tokens` or use CPU instead of GPU.
    * [ ] Optimize Camelot by limiting pages in `extract_tables_with_camelot` for large PDFs.
* [ ] **Data Validation**:
    * [ ] Validate `corrections/<pdf_id>_tasks.json` and `corrections/<pdf_id>.json` against expected schemas.
    * [ ] Debug JSON parsing errors in `app.py` (e.g., missing `annotations` field).

**IV. Infrastructure and Tooling:**

* [ ] **Docker Verification**:
    * [ ] Confirm `Dockerfile` installs all system dependencies (e.g., `tesseract-ocr`, `poppler-utils`).
    * [ ] Test `docker-compose.yml` with:
        ```bash
        docker compose down && docker compose up --build
        ```
    * [ ] Debug service startup failures (e.g., port conflicts, volume permissions).
* [ ] **Volume Management**:
    * [ ] Ensure volumes (`uploads`, `output`, `corrections`) are writable by both services.
    * [ ] Debug permission issues: `docker exec flask-api chmod -R 777 /app/corrections`.

**V. Code Quality and Standards:**

* [ ] **Code Style**: Adhere to PEP 8 guidelines.
* [ ] **Docstrings:**
    * [ ] Verify `app.py` and `pdf_to_json_converter.py` have module-level docstrings with purpose, third-party links, and sample input/output.
        * **Improved Instructions:**
            1. Open `app.py` and `pdf_to_json_converter.py` in a text editor.
            2.  Verify that each file has a docstring at the very beginning, enclosed in triple quotes (`"""Docstring goes here"""`).
            3.  Ensure the docstring includes:
                *   A brief description of the module's purpose (e.g., "This module implements the Flask API endpoints for the PDF extraction service.").
                *   Links to the documentation of any third-party packages used in the module (e.g., "[Camelot Documentation](https://camelot-py.readthedocs.io/)").
                *   If applicable, a simple example of how to use the module or its key functions, including sample input and expected output.
            4.  If a module-level docstring is missing or incomplete, add or update it accordingly.
    * [ ] Add missing docstrings for new functions (e.g., endpoint handlers in `app.py`).
        * **Improved Instructions:**
            1.  For each function or method in `app.py` and `pdf_to_json_converter.py` that does not have a docstring, add one.
            2.  The docstring should include:
                *   A brief description of the function's purpose.
                *   A description of the function's parameters, including their types and meanings.
                *   A description of the function's return value, including its type and meaning.
                *   If the function raises any exceptions, document them.
            3.  Follow the [PEP 257](https://peps.python.org/pep-0257/) docstring conventions.
        * **Automated Checkpoint:** (Tool Suggestion)
            1. Install pydocstyle
                ```bash
                pip install pydocstyle
                ```
            2. Run pydocstyle on the flask-api service
                ```bash
                docker exec flask-api pydocstyle app.py
                docker exec flask-api pydocstyle pdf_to_json_converter.py
                ```
            3. The command should return an empty output, or suggestions that can be implemented

* [ ] **Standalone Verification**:
    * [ ] Ensure `pdf_to_json_converter.py`’s `usage_example` runs independently:
        ```bash
        python pdf_to_json_converter.py
        ```
    * [ ] Add a similar block to `app.py` if missing, testing endpoints locally.
* [ ] **Logging**:
    * [ ] Confirm `loguru` logs critical events (e.g., extraction start, correction application, errors).
    * [ ] Add debug logs for merging and parameter tweaking if missing.
* [ ] Add new tools for code formatteing and testing

**VI. Testing Strategy:**

*   **Pre-Pytest Verification**:
    *   Only proceed to pytest after `app.py` and `pdf_to_json_converter.py` usage functions work as expected.
    *   Manually test each endpoint and function with test PDFs (text-based, scanned, multi-page tables).
*   **Manual Testing**:
    *   Run end-to-end workflow for each test case (text-based, scanned, parameter tweaking).
    *   Verify `structured.json` for correctness (e.g., merged tables, human edits).
*   **Pytest (Post-Verification)**:
    *   Create `tests/unit/test_pdf_to_json_converter.py` for functions like `merge_camelot_tables`, `extract_tables_with_camelot`.
    *   Create `tests/integration/test_app.py` for Flask endpoints.
    *   Run tests:
        ```bash
        docker exec flask-api pip install pytest
        docker exec flask-api pytest /app/tests
        ```

**VII. Agentic Workflow:**

*   **Planner**: Break down debugging tasks (e.g., verify Docker, test endpoints, debug merging).
*   **Coder**: Fix code issues (e.g., add error handling in `app.py`, optimize Qwen-VL-2B).
*   **Debugger**: Analyze logs, reproduce failures, and suggest fixes (e.g., dependency updates).
*   **Tester**: Run manual tests for each test case, verify outputs.
*   **Documenter**: Update `README.md` and docstrings with debugging insights.

**VIII. Success Criteria:**

*   Docker services start without errors.
*   `pdf_to_json_converter.py`’s `usage_example` produces correct `structured.json`.
*   Flask API endpoints handle PDF uploads, page serving, correction saving, and re-extraction.
*   LabelStudio loads tasks, supports annotations, and exports correct JSON.
*   End-to-end workflow completes for text-based, scanned, and parameter-tweaked PDFs.
*   Multi-page table merges (automated and human-directed) appear correctly in `structured.json`.
*   Corrections are applied, with approved/edited tables prioritized and rejected tables excluded.

**IX. Test Data:**
    * All test data will be located in ``` test_data/test_pdf ```
    *
        Add a simple PDF to the repository test_data folder.
Add a multi-page PDF to the repository test_data folder.
Add a Scanned PDF to the repository test_data folder.
Test data should include
    * Test data must include input
    * The input must follow with expected output file
    * The input must be catagorized
Test_data Example
* ``` test_data/
pdf_test/
simple.pdf - input
simple.json - outout.json
```

This is the complete and updated `task.md` file. I have incorporated your feedback and made several improvements to make it more actionable and agent-friendly.

Is there anything else I can do to improve this task plan? I am ready for your next set of instructions