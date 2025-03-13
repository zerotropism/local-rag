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
    Set default distance parameters to cosine.

    Args:
        vector_db (QdrantClient): vector database client.
        encoder (SentenceTransformer): encoder object.
        collection_name (str): name of the collection.

    Returns:
        int: 0 if successful.
    """
    vector_db.create_collection(
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
    """Serialize data into the vector database.Each data point is represented as a vector.

    Args:
        vector_db (QdrantClient): vector database client.
        target_collection_name (str): name of the collection to upload data in.
        data (List[Dict]): serialized data to be uploaded.
        encoder (SentenceTransformer): encoder object.

    Returns:
        int: 0 if successful.
    """
    vector_db.upload_points(
        collection_name=target_collection_name,
        points=[
            models.PointStruct(
                id=index,
                vector=encoder.encode(content["notes"]).tolist(),
                payload=content,
            )
            for index, content in enumerate(data)
        ],
    )
    return 0


def check_vector_db(
    vector_db: QdrantClient, collection_name: str, filter: str = None, value: str = None
) -> models.CollectionInfo:
    """Check if data is stored in the vector database.

    Args:
        vector_db (QdrantClient): vector database client.
        collection_name (str): name of the collection to search in.

    Returns:
        check (CollectionInfo): number of points in the collection.
    """
    return vector_db.get_collection(collection_name)


def search_vector_db(
    vector_db: QdrantClient,
    collection_name: str,
    encoder: SentenceTransformer,
    query: str,
    return_limit: int = 2,
) -> List[models.ScoredPoint]:
    """Search the vector database for similar embeddings. Deprecated use `query_points` instead of `search`.

    Args:
        vector_db (QdrantClient): vector database client.
        collection_name (str): name of the collection to search in.
        encoder (SentenceTransformer): encoder object.
        query (str): input user query.
        return_limit (int, optional): number of results to return. Defaults to 2.

    Returns:
        list: list of ScoredPoint objects.
    """
    hits = vector_db.search(
        collection_name=collection_name,
        query_vector=encoder.encode(query).tolist(),
        limit=return_limit,
    )
    return hits


def pprint_results(hits: List[models.ScoredPoint]):
    """Pretty print search results.

    Args:
        hits (List[ScoredPoint]): search results.
    """
    print("\n")
    print("-" * 130)
    print("Search results:", "\n")
    [
        print(
            hit.payload["name"],
            hit.payload["region"],
            "score:",
            str(round(float(hit.score), 3)),
        )
        for hit in hits
    ]
    print("-" * 130)
    print("\n")


if __name__ == "__main__":
    # load data
    data = load_data("top_rated_wines.csv")

    # create embeddings
    encoder = create_embeddings("all-MiniLM-L6-v2")

    # create the vector database client
    vdb = create_vector_db(":memory:")

    # create the collection
    new_collection_name = "top_wines"
    add_collection_to_vector_db(vdb, encoder, new_collection_name)

    # serialize data into vector database
    vectorize_data(vdb, new_collection_name, data, encoder)

    # check if data is stored in the vector database
    checkpoints = check_vector_db(vdb, new_collection_name)
    print(
        f"Collection {new_collection_name} contains {checkpoints.points_count} points."
    )

    # search locally
    user_query = "A historic French wine"
    hits = search_vector_db(vdb, new_collection_name, encoder, user_query, 3)

    # print results
    pprint_results(hits)
