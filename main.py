import pandas as pd
from typing import List, Dict
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer


def load_data(dataset: str = "dataset.csv") -> List[Dict]:
    """Load data from a CSV file.

    Args:
        dataset (str, optional): Defaults to "dataset.csv".

    Returns:
        List[Dict]: serialized data as a list of dictionaries.
    """
    df = pd.read_csv(dataset)
    df = df[df["variety"].notna()]  # remove any NaN values as it blows up serialization
    data = df.to_dict("records")
    return data


def create_embeddings(model_or_path: str) -> SentenceTransformer:
    """Create embeddings using SentenceTransformer object.

    Args:
        model_or_path (str): model name or path to the model used to create embeddings.

    Returns:
        SentenceTransformer: encoder object.
    """
    encoder = SentenceTransformer(model_or_path)
    return encoder


def create_vector_db(mode: str) -> QdrantClient:
    """Create a vector database with QdrantClient object.

    Args:
        mode (str): mode of the vector database, i.e. ":memory:" or "path/to/db".

    Returns:
        QdrantClient: vector database client.
    """
    vector_db = QdrantClient(mode)  # create in-memory Qdrant instance
    return vector_db


def add_collection_to_vector_db(
    vector_db: QdrantClient, encoder: SentenceTransformer, collection_name: str
):
    """Create a collection in the vector database where to upload embedded data later.
    Set default distance parameters to cosine. Deprecated: use `create_collection` instead.

    Args:
        vector_db (QdrantClient): vector database client.
        encoder (SentenceTransformer): encoder object.
        collection_name (str): name of the collection.

    Returns:
        int: 0 if successful.
    """
    vector_db.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),  # vector size is defined by used model
            distance=models.Distance.COSINE,
        ),
    )
    return 0


def vectorize_data(
    vector_db: QdrantClient,
    target_collection_name: str,
    data: List[Dict],
    encoder: SentenceTransformer,
):
    """Serialize data into the vector database.

    Args:
        vector_db (QdrantClient): vector database client.
        target_collection_name (str): name of the collection to upload data in.
        data (List[Dict]): serialized data to be uploaded.
        encoder (SentenceTransformer): encoder object.

    Returns:
        int: 0 if successful.
    """
    # deprecated: upload using "Points" instead of "Records" (see Qdrant docs)
    vector_db.upload_records(
        collection_name=target_collection_name,
        records=[
            models.Record(
                id=idx, vector=encoder.encode(doc["notes"]).tolist(), payload=doc
            )
            for idx, doc in enumerate(data)
        ],
    )
    return 0


def search_vector_db(
    vector_db: QdrantClient,
    collection_name: str,
    encoder: SentenceTransformer,
    query: str,
    return_limit: int = 2,
):
    """Search the vector database for similar embeddings.

    Args:
        vector_db (QdrantClient): vector database client.
        collection_name (str): name of the collection to search in.
        encoder (SentenceTransformer): encoder object.
        query (str): input user query.
        return_limit (int, optional): number of results to return. Defaults to 2.

    Returns:
        list: list of hits.
    """
    hits = vector_db.search(
        collection_name=collection_name,
        query_vector=encoder.encode(query).tolist(),
        limit=return_limit,
    )
    return hits


def scroll_vector_db(
    vector_db: QdrantClient, collection_name: str, filter: str, value: str
):
    """Naive check if embeddings are stored in the vector database.
    Deprecated: use `collection_exists` instead.

    Args:
        vector_db (QdrantClient): vector database client.
        collection_name (str): name of the collection to search in.
        filter (str): key field condition to filter by.
        value (str): value to filter by.

    Returns:
        list: list of results. Limits to 3 results.
    """
    results = vector_db.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key=filter, match=models.MatchValue(value=value)),
            ]
        ),
        limit=3,
        with_payload=True,
        with_vectors=False,
    )
    return results[0]


if __name__ == "__main__":
    # load data
    data = load_data("top_rated_wines.csv")

    # create embeddings
    encoder = create_embeddings("all-MiniLM-L6-v2")

    # create the vector database client
    vdb = create_vector_db(":memory:")

    # create the collection
    add_collection_to_vector_db(vdb, encoder, "top_wines")

    # serialize data into vector database
    vectorize_data(vdb, "top_wines", data, encoder)

    # check if embeddings are properly stored
    if scroll_vector_db(vdb, "top_wines", "variety", "Red Wine"):
        print("Data have been properly stored.")

        # search locally
        hits = search_vector_db(vdb, "top_wines", encoder, "A historic French wine", 3)

        # print results
        print("-" * 90)
        print("\n", "Search results:", "\n")
        for hit in hits:
            print(hit.payload["name"], "score:", hit.score)
        print("-" * 90)

    else:
        print("No data found.")
