from pathlib import Path

import pandas as pd
import yaml
from chatbot import Chatbot
from vectordb import VectorDB


def load_configurations(path: str = "code/config.yaml") -> dict:
    """Load configurations from a YAML file.
    Args:
        path (str, optional): path to the YAML file. Defaults to "config.yaml".
    Returns:
        Dict: configurations as a dictionary.
    """
    with Path(path).open() as file:
        conf = yaml.safe_load(file)
    return conf


def load_data(path: str) -> list[dict]:
    """Load data from a CSV file.
    Returns:
        List[Dict]: serialized data as a list of dictionaries.
    """
    df = pd.read_csv(path)
    # remove any NaN values as it blows up serialization
    df = df[df["variety"].notna()]
    data = df.to_dict("records")
    return data


def setup_vector_db(conf: dict) -> VectorDB:
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


def setup_chatbot(conf: dict, vector_db: VectorDB):
    """Setup the chatbot.
    Args:
        conf (Dict): chatbot related configurations.
        user_query (str): user initial query.
        hits (List): vector db search results from initial user query.
    Returns:
        Chatbot: chatbot object.
    """
    # check if chatbot configurations are provided
    if not conf:
        print(
            "Chatbot configurations not found. "
            "Please provide chatbot configurations in the config file."
        )
        return
    # instantiate the chatbot object
    bot = Chatbot(
        theme=conf.get("theme"),
        template=conf.get("initial_template"),
        model=conf.get("llm_model"),
        rag=vector_db,
    )
    return bot


def main():
    # load conf
    conf = load_configurations("code/config.yaml")

    # setup vector db
    vdb = setup_vector_db(conf)

    # setup chatbot
    bot = setup_chatbot(conf.get("chatbot", {}), vdb)

    # start conversation
    bot.handle_conversation()


if __name__ == "__main__":
    main()
