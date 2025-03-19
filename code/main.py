import yaml
import pandas as pd
from typing import List, Dict
from vectordb import VectorDB
from chatbot import Chatbot


def load_configurations(path: str = "code/config.yaml") -> Dict:
    """Load configurations from a YAML file.

    Args:
        path (str, optional): path to the YAML file. Defaults to "config.yaml".

    Returns:
        Dict: configurations as a dictionary.
    """
    with open(path, "r") as file:
        conf = yaml.safe_load(file)
    return conf


def load_data(path: str) -> List[Dict]:
    """Load data from a CSV file.

    Returns:
        List[Dict]: serialized data as a list of dictionaries.
    """
    df = pd.read_csv(path)
    # remove any NaN values as it blows up serialization
    df = df[df["variety"].notna()]
    data = df.to_dict("records")
    return data


def setup_vector_db(conf: Dict) -> VectorDB:
    """Load the vector database.

    Args:
        conf (Dict): vector db related configurations.

    Returns:
        VectorDB: vectordb object.
    """
    data_path = conf.get("data")
    vectordb_conf = conf.get("vectordb")

    # instantiate the vectordb object
    vdb = VectorDB(
        datapoints=load_data(data_path),
        encoder_model=vectordb_conf.get("encoder_model"),
        instance_mode=vectordb_conf.get("instance_mode"),
        collection_name=vectordb_conf.get("collection_name"),
    )

    # build the vectordb client
    vdb.build()

    return vdb


def pprint_results(user_query: str, hits: List):
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


def setup_chatbot(conf: Dict, user_query: str, hits: List):
    """Setup the chatbot.

    Args:
        conf (Dict): chatbot related configurations.
        user_query (str): user initial query.
        hits (List): vector db search results from initial user query.

    Returns:
        Chatbot: chatbot object.
    """
    # instantiate the chatbot object
    bot = Chatbot(
        question=user_query,
        search=hits,
        theme=conf.get("theme"),
        template=conf.get("initial_template"),
        model=conf.get("llm_model"),
    )
    return bot


def main():
    # load conf
    conf = load_configurations("code/config.yaml")

    # setup vector db
    vdb = setup_vector_db(conf)

    # collect user query
    user_query = input("Enter your query: ")

    # search locally
    hits = vdb.search_vector_db(query=user_query, return_limit=3)

    # print results
    pprint_results(user_query, hits)

    # setup chatbot
    bot = setup_chatbot(conf.get("chatbot", {}), user_query, hits)

    # start conversation
    bot.start_conversation()


if __name__ == "__main__":
    main()
