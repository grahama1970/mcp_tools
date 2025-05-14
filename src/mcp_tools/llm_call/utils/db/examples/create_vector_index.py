from loguru import logger

def create_vector_index(db, collection_name: str, dimension: int = 384):
    """Ensure a vector index exists on the specified collection."""
    if not db.collection(collection_name).has_index("vector_cosine"):
        db.collection(collection_name).add_index({
            "type": "vector",
            "fields": ["vector_data"],
            "params": {
                "metric": "cosine",
                "dimension": dimension
            }
        })
        logger.info(f"Vector index created for '{collection_name}'.")
    else:
        logger.info(f"Vector index already exists for '{collection_name}'.")
