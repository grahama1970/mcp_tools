from arango import ArangoClient
from loguru import logger
from typing import List, Dict, Any, Optional
# Assuming embedding_utils is structured correctly for relative import
# If it's in mcp_litellm/utils/, the import might need adjustment based on execution context
# Let's assume it's accessible as below for now.
try:
    from .. import embedding_utils # Relative import if arango_utils is called from within mcp_litellm package
except ImportError:
    # Fallback for direct script execution or different structure
    import sys
    from pathlib import Path
    # Add the parent directory of 'utils' to sys.path if needed
    utils_dir = Path(__file__).parent
    src_dir = utils_dir.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from mcp_litellm.utils import embedding_utils



def connect_to_arango_client(config):
    """Connect to the ArangoDB client, create database if it does not exist."""
    client = ArangoClient(hosts=config['host'])
    db_name = config['database_name']
    username = config['username']
    password = config['password']

    # Connect to the _system database to manage databases
    sys_db = client.db('_system', username=username, password=password)

    # Check if target database exists
    if not sys_db.has_database(db_name):
        logger.info(f"Database '{db_name}' does not exist. Creating it.")
        sys_db.create_database(db_name)
    else:
        logger.info(f"Database '{db_name}' already exists.")

    # Connect to the target database
    return client.db(db_name, username=username, password=password, verify=True)

def insert_object(db, obj):
    """Insert a single STIX object into the database."""
    collection_name = obj.get('type')
    collection = db.collection(collection_name)
    obj['_key'] = obj['id']
    collection.insert(obj, overwrite=True)

def handle_relationships(db, stix_objects):
    """Process and create edge relationships for STIX objects."""
    skipped_relationships = []
    if not db.has_collection('relationships'):
        db.create_collection('relationships', edge=True)
    edge_collection = db.collection('relationships')
    for obj in stix_objects:
        if obj.get('type') == 'relationship':
            (source_id, target_id) = (obj.get('source_ref'), obj.get('target_ref'))
            if not source_id or not target_id:
                skipped_relationships.append(obj['id'])
                continue
            edge_document = {'_from': f"{source_id.split('--')[0]}/{source_id}", '_to': f"{target_id.split('--')[0]}/{target_id}", '_key': obj['id'], 'relationship_type': obj.get('relationship_type', 'unknown')}
            edge_collection.insert(edge_document, overwrite=True)
    if skipped_relationships:
        logger.warning(f'Skipped relationships: {skipped_relationships}')


def get_lessons(db, role=None, category=None, identifier=None):
    """
    Query lessons from the 'lessons_learned' collection with optional filters.
    Returns a list of lesson documents.
    """
    filters = []
    bind_vars = {}

    if role:
        filters.append("lesson.role == @role")
        bind_vars["role"] = role
    if category:
        filters.append("lesson.category == @category")
        bind_vars["category"] = category
    if identifier:
        filters.append("lesson.identifier == @identifier")
        bind_vars["identifier"] = identifier

    filter_clause = ""
    if filters:
        filter_clause = "FILTER " + " AND ".join(filters)

    query = f"""
    FOR lesson IN lessons_learned
        {filter_clause}
        RETURN lesson
    """

    cursor = db.aql.execute(query, bind_vars=bind_vars)
    return [doc for doc in cursor]


def upsert_lesson(db, lesson: Dict[str, Any]):
    """
    Upsert a lesson into the 'lessons_learned' collection.
    Creates or replaces the lesson based on composite key.
    Also generates and stores a vector embedding for the lesson text.
    """
    collection = db.collection("lessons_learned")
    key = f"{lesson.get('role','')}_{lesson.get('category','')}_{lesson.get('identifier','')}".replace(" ", "_")
    lesson['_key'] = key

    # Generate and add embedding
    lesson_text = lesson.get('lesson')
    if lesson_text:
        try:
            # Use the synchronous version if running in sync context, or manage async appropriately
            embedding_data = embedding_utils.create_embedding_sync(lesson_text)
            lesson['lesson_embedding'] = embedding_data.get('embedding')
            # Optionally store embedding metadata too
            # lesson['embedding_metadata'] = embedding_data.get('metadata')
            logger.info(f"Generated embedding for lesson: {key}")
        except Exception as e:
            logger.error(f"Failed to generate embedding for lesson {key}: {e}")
            lesson['lesson_embedding'] = None # Ensure field exists even if generation fails

    try:
        collection.insert(lesson, overwrite=True)
        logger.info(f"Upserted lesson: {key}")
    except Exception as e:
        logger.warning(f"Upsert failed for lesson {key}: {e}")

def update_lesson(db, lesson: Dict[str, Any]):
    """
    Update an existing lesson in the 'lessons_learned' collection.
    Requires the same composite key logic as upsert.
    Updates the embedding if the lesson text changes.
    """
    collection = db.collection("lessons_learned")
    key = f"{lesson.get('role','')}_{lesson.get('category','')}_{lesson.get('identifier','')}".replace(" ", "_")
    lesson['_key'] = key

    # Check if lesson text exists and potentially generate/update embedding
    lesson_text = lesson.get('lesson')
    update_payload = lesson.copy() # Work on a copy to avoid modifying input dict directly for embedding logic
    should_update_embedding = False
    if 'lesson' in update_payload: # Only update embedding if lesson text is part of the update
        should_update_embedding = True
        if not lesson_text: # Handle case where lesson text is explicitly set to empty/null
             update_payload['lesson_embedding'] = None
             logger.info(f"Removing embedding for updated lesson (empty text): {key}")


    if should_update_embedding and lesson_text:
        try:
            embedding_data = embedding_utils.create_embedding_sync(lesson_text)
            update_payload['lesson_embedding'] = embedding_data.get('embedding')
            logger.info(f"Generated embedding for updated lesson: {key}")
        except Exception as e:
            logger.error(f"Failed to generate embedding for updated lesson {key}: {e}")
            # Decide how to handle embedding update failure - keep old one? set to null?
            # Let's set to null to indicate failure during update.
            update_payload['lesson_embedding'] = None


    try:
        # Use update=True to only update specified fields
        # Use keep_none=False to remove fields set to None (like failed embedding)
        # Use merge_objects=True to merge sub-objects if any
        # Pass the potentially modified update_payload
        collection.update(update_payload, merge_objects=True, keep_none=False)
        logger.info(f"Updated lesson: {key}")
    except Exception as e:
        logger.warning(f"Update failed for lesson {key}: {e}")



def delete_lesson(db, role, category, identifier):
    """
    Delete a lesson from the 'lessons_learned' collection by composite key.
    """
    collection = db.collection("lessons_learned")
    key = f"{role}_{category}_{identifier}".replace(" ", "_")

    try:
        collection.delete(key)
    except Exception as e:
        logger.warning(f"Delete failed: {e}")


# --- New Agent Query Functions ---

def query_lessons_by_keyword(db, keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Query lessons by keywords, searching in lesson text, category, and identifier.
    Uses simple case-insensitive text matching (LIKE).

    Args:
        db: ArangoDB database connection object.
        keywords: A list of keywords to search for.
        limit: Maximum number of results to return.

    Returns:
        A list of matching lesson documents.
    """
    if not keywords:
        return []

    # Build filter conditions for each keyword across multiple fields
    keyword_filters = []
    bind_vars = {}
    for i, keyword in enumerate(keywords):
        # Basic sanitization: remove potential AQL injection characters?
        # For LIKE, '%' and '_' are wildcards. Escape them if searching literally?
        # Assuming keywords are simple words for now.
        safe_keyword = str(keyword).replace('%', '\\%').replace('_', '\\_')
        key_var = f"keyword{i}"
        bind_vars[key_var] = f"%{safe_keyword}%" # Prepare for LIKE with wildcards around the keyword
        keyword_filters.append(
            f"""(
                LIKE(LOWER(lesson.lesson), LOWER(@{key_var}), true) OR
                LIKE(LOWER(lesson.category), LOWER(@{key_var}), true) OR
                LIKE(LOWER(lesson.identifier), LOWER(@{key_var}), true)
            )"""
        ) # Using LIKE for case-insensitive partial match

    # Combine keyword filters with OR (lesson matches any keyword)
    filter_clause = "FILTER " + " OR ".join(keyword_filters)

    query = f"""
    FOR lesson IN lessons_learned
        {filter_clause}
        LIMIT @limit
        RETURN lesson
    """
    bind_vars["limit"] = limit

    try:
        cursor = db.aql.execute(query, bind_vars=bind_vars)
        return [doc for doc in cursor]
    except Exception as e:
        logger.error(f"Keyword query failed: {e}")
        return []

def query_lessons_by_concept(db, concepts: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Query lessons by concepts (currently implemented as keyword search).
    Searches concept terms in lesson text, category, and identifier.

    Args:
        db: ArangoDB database connection object.
        concepts: A list of concept terms to search for.
        limit: Maximum number of results to return.

    Returns:
        A list of matching lesson documents.
    """
    # Currently, this is an alias for keyword search.
    # Future enhancement: Use NLP techniques for true concept matching.
    logger.info("Executing concept query (using keyword search implementation).")
    return query_lessons_by_keyword(db, concepts, limit)


def query_lessons_by_similarity(db, query_text: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Query lessons by semantic similarity to a given text.
    Requires lessons to have 'lesson_embedding' field and assumes a vector index
    (e.g., named 'idx_lesson_embedding' using cosine similarity) is configured on that field.

    Args:
        db: ArangoDB database connection object.
        query_text: The text to find similar lessons for.
        top_n: The number of most similar lessons to return.

    Returns:
        A list of the top N most similar lesson documents, ordered by similarity.
        Returns empty list on failure or if prerequisites are not met.
    """
    try:
        # 1. Generate embedding for the query text
        query_embedding_data = embedding_utils.create_embedding_sync(query_text)
        query_vector = query_embedding_data.get('embedding')
        if not query_vector or not isinstance(query_vector, list):
            logger.error(f"Failed to generate valid embedding for query text: {query_text}")
            return []
        if len(query_vector) != embedding_utils.DEFAULT_EMBEDDING_DIM:
             logger.warning(f"Query vector dimension mismatch: expected {embedding_utils.DEFAULT_EMBEDDING_DIM}, got {len(query_vector)}")
             # Proceeding anyway, but similarity might be meaningless

        # 2. Execute AQL vector search query
        # This query uses manual cosine similarity calculation.
        # Replace with ArangoDB's native vector search functions if an index is available.
        # Example using COSINE_SIMILARITY (requires ArangoDB 3.11+ and appropriate index):
        # query = f"""
        # LET queryVector = @query_vector
        # FOR lesson IN lessons_learned
        #     FILTER HAS(lesson, 'lesson_embedding') AND lesson.lesson_embedding != null
        #     LET similarity = COSINE_SIMILARITY(lesson.lesson_embedding, queryVector)
        #     FILTER similarity > 0 // Optional: Filter out dissimilar results
        #     SORT similarity DESC
        #     LIMIT @top_n
        #     RETURN MERGE(lesson, {{ similarity_score: similarity }})
        # """
        # Using manual calculation as a fallback:
        # Use AQL COSINE_SIMILARITY with vector index
        query = """
        FOR doc IN lessons_learned
        LET similarity_score = COSINE_SIMILARITY(doc.lesson_embedding, @query_vector)
        SORT similarity_score DESC
        LIMIT @top_n
        RETURN {
          document: doc,
          similarity_score: similarity_score
        }
        """

        bind_vars = {
            "query_vector": query_vector,
            "top_n": top_n
            # "expected_dim": embedding_utils.DEFAULT_EMBEDDING_DIM # No longer needed in COSINE_SIMILARITY query
        }

        cursor = db.aql.execute(query, bind_vars=bind_vars)
        results = [doc for doc in cursor]
        logger.info(f"Found {len(results)} similar lessons for query '{query_text[:50]}...'.")
        return results

    except Exception as e:
        # Log the specific AQL error if possible
        aql_error_num = getattr(e, 'http_exception', {}).get('errorNum', None)
        if aql_error_num:
             logger.error(f"Similarity query failed with AQL error {aql_error_num}: {e}")
        else:
             logger.error(f"Similarity query failed: {e}")
        return []