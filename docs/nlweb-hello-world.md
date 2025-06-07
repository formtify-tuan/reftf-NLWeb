==# NLWeb Hello World

## Getting Started

Getting your NLWeb Server up and running locally.

This will get you up and running using a local vector database. You can load RSS feeds, legal documents or any corporate docs to test the search pipeline.

## Prerequisites

These instructions assume that you have Python 3.10+ installed locally.

## From the Terminal

1. Clone or download the code from the repo.

    ```sh
    git clone https://github.com/microsoft/NLWeb
    cd NLWeb
    ```

2. Open a terminal to create a virtual python environment and activate it.

    ```sh
    python -m venv myenv
    source myenv/bin/activate    # Or on Windows: myenv\Scripts\activate
    ```

3. Go to the 'code' folder in NLWeb to install the dependencies. Note that this will also install the local vector database requirements so you don't need to install them separately.

    ```sh
    cd code
    pip install -r requirements.txt
    ```

4. Copy the .env.template file to a new .env file and update the API key you will use for your LLM endpoint of choice. The local Qdrant database variables are already set for this exercise.  Don't worry; you do not need to provide all of these providers in the file.  We explain below.

    ```sh
    cp .env.template .env
    ```

5. Update your config files (located in the code/config folder) to make sure your preferred providers match your .env file. There are three files that may need changes.

    - config_llm.yaml: Set the preferred endpoint to Azure OpenAI and make sure your environment variables match.
    - config_embedding.yaml: Set the preferred provider to Azure OpenAI.
    - config_retrieval.yaml: Use `qdrant_local` for a local database or `qdrant_url` to connect to a remote Qdrant instance.

6. Now we will load some data in our local vector database. You can load RSS feeds as before or point the new `llamaindex_demo.py` script at a folder of legal or corporate documents in DOCX format. The script will create chunks using LlamaIndex and store them in Qdrant.

    The format of the command is as follows (make sure you are still in the 'code' folder when you run this):

    ```sh
    python -m tools.db_load <RSS URL> <site-name>
    ```

    Kevin's 'Behind the Tech' Podcast:

    ```sh
    python -m tools.db_load https://feeds.libsyn.com/121695/rss Behind-the-Tech
    ```

    Verge's 'Decoder' Podcast:

    ```sh
    python -m tools.db_load https://feeds.megaphone.fm/recodedecode Decoder
    ```

    You can find even more data, including other formats other than RSS, in this [OneDrive folder](https://1drv.ms/f/c/6c6197aa87f7f4c4/EsT094eql2EggGxlBAAAAAABajQiZ5unf_Ri_OWksR8eNg?e=I4z5vw). (Note:  If it asks you to login, try the URL a 2nd time. It should be open permissions.)

    For corporate documents you can run the new demo script from the repo root:

    ```sh
    python tools/llamaindex_demo.py --docs ./docs
    ```

    The script loads DOCX files, splits them into ~200–400 token chunks, stores the embeddings in Qdrant,
    and starts an interactive prompt powered by GPT‑4o‑mini.

7. Start your NLWeb server (again from the 'code' folder):

    ```sh
    python app-file.py
    ```

8. Go to localhost:8000/

9. You should have a working search!  You can also try different sample UIs by adding 'static/<html file name>' to your localhost path above.
