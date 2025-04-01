from typing import List, Dict, Union
from decorators import handle_exception
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
        """Initialize the VectorDB class.
        Args:
            datapoints (List[Dict]): List of dictionaries containing data points.
            encoder_model (str, optional): Name of the encoder model to be used. Defaults to "all-MiniLM-L6-v2".
            instance_mode (str, optional): Mode for the QdrantClient instance. Defaults to ":memory:".
            collection_name (str, optional): Name of the collection in the vector database. Defaults to "new_collection".
        """
        self._encoder_model = encoder_model
        self._instance_mode = instance_mode
        self._collection_name = collection_name
        self._data = datapoints
        self._encoder = None
        self._vector_db = None
        self._checkpoints = None

    @property
    def encoder_model(self) -> str:
        """Get the currently used encoder model name."""
        return self._encoder_model

    @encoder_model.setter
    def encoder_model(self, model_name: str):
        """Set the encoder model name to be used for embedding generation.

        Args:
            model_name (str): Name of the encoder model to be used.
        """
        self._encoder_model = model_name
        self._encoder = SentenceTransformer(model_name)

    @property
    def collection_name(self) -> str:
        """Get the currently used collection name."""
        return self._collection_name

    @collection_name.setter
    def collection_name(self, name: str):
        """Set the collection name to be used for the vector database.

        Args:
            name (str): Name of the collection to be used.
        """
        self._collection_name = name

    @handle_exception
    def create_embeddings(self) -> SentenceTransformer:
        """Create embeddings using SentenceTransformer object.
        Returns:
            SentenceTransformer: Encoder object.
        """
        self._encoder = SentenceTransformer(self._encoder_model)
        return self._encoder

    @handle_exception
    def create_vector_db(self) -> QdrantClient:
        """Create a vector database of a QdrantClient class with set instance mode.
        Returns:
            QdrantClient: Vector database client.
        """
        self._vector_db = QdrantClient(self._instance_mode)
        return self._vector_db

    @handle_exception
    def add_collection_to_vector_db(self) -> int:
        """Create a collection in the vector database where to upload embedded data later.
        Default distance parameter is set to cosine.
        Returns:
            int: 0 if successful.
        """
        self._vector_db.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                # vector size is defined by used model
                size=self._encoder.get_sentence_embedding_dimension(),
                # default distance is cosine
                distance=models.Distance.COSINE,
            ),
        )
        return 0

    @handle_exception
    def vectorize_data(self) -> int:
        """Serialize data into the vector database. Each data point is represented as a vector.
        Returns:
            int: 0 if successful.
        """
        self._vector_db.upload_points(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=index,
                    vector=self._encoder.encode(content["notes"]).tolist(),
                    payload=content,
                )
                for index, content in enumerate(self._data)
            ],
        )
        return 0

    @handle_exception
    def check_vector_db(self) -> models.CollectionInfo:
        """Check if data is stored in the vector database.
        Returns:
            models.CollectionInfo: Collection information including number of corresponding datapoints.
        """
        self._checkpoints = self._vector_db.get_collection(self._collection_name)
        return self._checkpoints

    @handle_exception
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
        if self._checkpoints.points_count:
            print("The vectordb has been set up successfully.")
            print(
                f"Collection '{self._collection_name}' contains {self._checkpoints.points_count} points."
            )
        else:
            print(
                "The vectordb has not been set up successfully: no data points found."
            )
        return 0

    @staticmethod
    def dict_to_indented_text(nested_dict: Dict) -> str:
        """Convert a nested dictionary to an indented text format.
        Args:
            nested_dict (Dict): Input nested dictionary.
        Returns:
            str: Indented text representation of the nested dictionary.
        """
        result = []
        for key, value in nested_dict.items():
            result.append(f"{key}:")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    result.append(f"\t{sub_key}: {sub_value}")
        return "\n".join(result)

    @handle_exception
    def search_vector_db(
        self, query: str, return_limit: int = 2, text_output: bool = False
    ) -> Union[str, Dict]:
        """Search the vector database for similar embeddings and returns a dictionnary of results that
        can optinally be serialized as plain text. Deprecated use `query_points` instead of `search`.
        Args:
            query (str): input user query.
            return_limit (int, optional): number of results to return. Defaults to 2.
        Returns:
            str: indented text representation of the search results if `text_output` is True.
            Dict: dictionary of search results.
        """
        search_results = {
            hit.payload["name"]: {
                "score": str(round(float(hit.score), 3)),
                "region": hit.payload["region"],
                "notes": hit.payload["notes"],
            }
            for hit in self._vector_db.search(
                collection_name=self._collection_name,
                query_vector=self._encoder.encode(query).tolist(),
                limit=return_limit,
            )
        }
        if text_output:
            return self.dict_to_indented_text(search_results)
        else:
            return search_results
