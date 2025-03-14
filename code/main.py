from typing import List
from vectordb import VectorDB


def pprint_results(hits: List):
    """Pretty print search results.

    Args:
        hits (List[ScoredPoint]): search results.
    """
    print("\n")
    print(f"Search results for '{user_query}':", "\n")
    [
        print(
            hit.payload["name"],
            hit.payload["region"],
            "score:",
            str(round(float(hit.score), 3)),
        )
        for hit in hits
    ]
    print("\n")


if __name__ == "__main__":

    # instantiate the vectordb client
    vdb = VectorDB(dataset="../data/top_rated_wines.csv", collection_name="top_wines")

    # build the vectordb
    vdb.build()

    # search locally
    user_query = "Suggest me an amazing wine from Saint Emilion"
    hits = vdb.search_vector_db(query=user_query, return_limit=3)

    # print results
    pprint_results(hits)
