from typing import List, Dict
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer


class VectorDB:
    def __init__(
        self,
        datapoints: List[Dict],
        encoder_model: str = "all-MiniLM-L6-v2",
        instance_mode: str = ":memory:",
        collection_name: str = "new_collection",
    ):
        self.encoder_model = encoder_model
        self.instance_mode = instance_mode
        self.collection_name = collection_name
        self.data = datapoints
        self.encoder = None
        self.vector_db = None
        self.checkpoints = None

    def create_embeddings(self) -> SentenceTransformer:
        """Create embeddings using SentenceTransformer object.

        Returns:
            SentenceTransformer: encoder object.
        """
        self.encoder = SentenceTransformer(self.encoder_model)
        return self.encoder

    def create_vector_db(self) -> QdrantClient:
        """Create a vector database with QdrantClient object.

        Returns:
            QdrantClient: vector database client.
        """
        # create in-memory Qdrant instance
        self.vector_db = QdrantClient(self.instance_mode)
        return self.vector_db

    def add_collection_to_vector_db(self) -> int:
        """Create a collection in the vector database where to upload embedded data later.
        Set default distance parameters to cosine.

        Returns:
            int: 0 if successful.
        """
        self.vector_db.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                # vector size is defined by used model
                size=self.encoder.get_sentence_embedding_dimension(),
                # default distance is cosine
                distance=models.Distance.COSINE,
            ),
        )
        return 0

    def vectorize_data(self) -> int:
        """Serialize data into the vector database. Each data point is represented as a vector.

        Returns:
            int: 0 if successful.
        """
        self.vector_db.upload_points(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=index,
                    vector=self.encoder.encode(content["notes"]).tolist(),
                    payload=content,
                )
                for index, content in enumerate(self.data)
            ],
        )
        return 0

    def check_vector_db(self) -> models.CollectionInfo:
        """Check if data is stored in the vector database.

        Returns:
            check (CollectionInfo): number of points in the collection.
        """
        self.checkpoints = self.vector_db.get_collection(self.collection_name)
        return self.checkpoints

    def build(self) -> int:
        """Build the vector database with the provided class instance parameters.

        Returns:
            int: 0 if successful.
        """
        # create embeddings
        self.create_embeddings()

        # create the vector database client
        self.create_vector_db()

        # create the collection
        self.add_collection_to_vector_db()

        # serialize data into vector database
        self.vectorize_data()

        # check if data is stored in the vector database
        self.check_vector_db()
        if self.checkpoints.points_count:
            print("The vectordb has been set up successfully.")
            print(
                f"Collection '{self.collection_name}' contains {self.checkpoints.points_count} points."
            )
        else:
            return print(
                "The vectordb has not been set up successfully: no data points found."
            )

    def search_vector_db(
        self, query: str, return_limit: int = 2
    ) -> List[models.ScoredPoint]:
        """Search the vector database for similar embeddings. Deprecated use `query_points` instead of `search`.

        Args:
            collection_name (str): name of the collection to search in.
            query (str): input user query.
            return_limit (int, optional): number of results to return. Defaults to 2.

        Returns:
            list: list of ScoredPoint objects.
        """
        hits = self.vector_db.search(
            collection_name=self.collection_name,
            query_vector=self.encoder.encode(query).tolist(),
            limit=return_limit,
        )
        return hits
