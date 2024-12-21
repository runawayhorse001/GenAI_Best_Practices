
.. _rag:

==============================
Retrieval-Augmented Generation
==============================

.. admonition:: Colab Notebook for This Chapter

    - Naive Chunking: |Naive Chunking|
    - Late Chunking: |Late Chunking|
    - Reciprocal Rank Fusion: |RRF|
    - RAG in colab: |n-rag|

    -------------

    - Self-RAG: |S-RAG|
    - Corrective RAG: |C-RAG|
    - Adaptive RAG: |A-RAG|
    - Agentic RAG: |Agentic-RAG|


    
    
    .. |n-rag| image:: images/colab-badge.png 
        :target: https://colab.research.google.com/drive/1jTkoER5eYQqBZs8RgOiCmTl8_WDt67go?usp=drive_link  

    .. |Naive Chunking| image:: images/colab-badge.png 
        :target: https://colab.research.google.com/drive/1r89QGmEDAS-ZJcfgYv6GKRrNDs4s_lir?usp=drive_link 

    .. |Late Chunking| image:: images/colab-badge.png 
        :target: https://colab.research.google.com/drive/1ZqxmK1YuvPcpJap1psscMTih2jnQVINT?usp=drive_link 

    .. |RRF| image:: images/colab-badge.png 
        :target: https://colab.research.google.com/drive/1Vg7C5Z3Y7OlilnfQlQqkKyNrcnUR6Xry?usp=drive_link 

    .. |S-RAG| image:: images/colab-badge.png 
        :target: https://colab.research.google.com/drive/1KbC4Wu-JUj6zbVuFkSNvtz3oVzWhEAZG?usp=drive_link 

    .. |C-RAG| image:: images/colab-badge.png 
        :target: https://colab.research.google.com/drive/1Sb7w1cTRZGkFs2LeF3F6gL8WNhjCdWyR?usp=drive_link 

    .. |A-RAG| image:: images/colab-badge.png 
        :target: https://colab.research.google.com/drive/1tzihPk9Ln96RYcwn6WaHHQjlMdFvCviS?usp=drive_link 

    .. |Agentic-RAG| image:: images/colab-badge.png 
        :target: https://colab.research.google.com/drive/1-eZ61ux0QSCNx2jUDq_vL-6OKLW8Hdr4?usp=drive_link 


.. _fig_rag:
.. figure:: images/RAG_diagram.png
    :align: center

    Retrieval-Augmented Generation Diagram


.. note::

  The naive chunking strategy was used in the diagram above. More advanced strategies, 
  such as Late Chunking [lateChunking]_ (or Chunked Pooling), are discussed later in this chapter.

Overview
++++++++

Retrieval-Augmented Generation (RAG) is a framework that enhances large language models (LLMs) 
by combining their generative capabilities with external knowledge retrieval. The goal of RAG 
is to improve accuracy, relevance, and factuality by providing the LLM with specific, up-to-date, 
or domain-specific context from a knowledge base or database during the generation process.

As you can see in :ref:`fig_rag`, the RAG has there main  components

- Indexer: The indexer processes raw text or other forms of unstructured data and creates an 
  efficient structure (called an index) that allows for fast and accurate retrieval 
  by the retriever when a query is made.

- Retriever: Responsible for finding relevant information from an external knowledge source, 
  such as a document database, a vector database, or the web.

- Generator: An LLM (like GPT-4, T5, or similar) that uses the retrieved context to generate a response.
  The model is "augmented" with the retrieved information, which reduces hallucination and enhances factual accuracy.

Indexing
++++++++

The indexing processes raw text or other forms of unstructured data and creates an efficient structure 
(called an index) that allows for fast and accurate retrieval by the retriever when a query is made.

.. _fig_indexing:
.. figure:: images/indexer.png
    :align: center

Naive Chunking
--------------

Chunking in Retrieval-Augmented Generation (RAG) involves splitting documents or knowledge bases 
into smaller, manageable pieces (chunks) that can be efficiently retrieved and used by a language model (LLM).

Below are the common chunking strategies used in RAG workflows:

1. **Fixed-Length Chunking**


   - Chunks are created with a predefined, fixed length (e.g., 200 words or 512 tokens).
   - Simple and easy to implement but might split content mid-sentence or lose semantic coherence.

   **Example**:

   .. code:: python

      def fixed_length_chunking(text, chunk_size=200):
          words = text.split()
          return [
              " ".join(words[i:i + chunk_size])
              for i in range(0, len(words), chunk_size)
          ]

      # Example Usage
      document = "This is a sample document with multiple sentences to demonstrate fixed-length chunking."
      chunks = fixed_length_chunking(document, chunk_size=10)
      for idx, chunk in enumerate(chunks):
          print(f"Chunk {idx + 1}: {chunk}")

   **Output**:
      
      - Chunk 1: This is a sample document with multiple sentences to demonstrate
      - Chunk 2: fixed-length chunking.

2. **Sliding Window Chunking**

   - Creates overlapping chunks to preserve context across splits.
   - Ensures important information in overlapping regions is retained.

   **Example**:

   .. code:: python

      def sliding_window_chunking(text, chunk_size=100, overlap_size=20):
          words = text.split()
          chunks = []
          for i in range(0, len(words), chunk_size - overlap_size):
              chunk = " ".join(words[i:i + chunk_size])
              chunks.append(chunk)
          return chunks

      # Example Usage
      document = "This is a sample document with multiple sentences to demonstrate sliding window chunking."
      chunks = sliding_window_chunking(document, chunk_size=10, overlap_size=3)
      for idx, chunk in enumerate(chunks):
          print(f"Chunk {idx + 1}: {chunk}")

   **Output**:
      - Chunk 1: This is a sample document with multiple sentences to demonstrate
      - Chunk 2: with multiple sentences to demonstrate sliding window chunking.
      - Chunk 3: sliding window chunking.

3. **Semantic Chunking**
  
   - Splits text based on natural language boundaries such as paragraphs, sentences, or specific delimiters (e.g., headings).
   - Retains semantic coherence, ideal for better retrieval and generation accuracy.

   **Example**:

   .. code:: python

      import nltk
      nltk.download('punkt_tab')

      def semantic_chunking(text, sentence_len=50):
          sentences = nltk.sent_tokenize(text)

          chunks = []
          chunk = ""
          for sentence in sentences:
              if len(chunk.split()) + len(sentence.split()) <= sentence_len:
                  chunk += " " + sentence
              else:
                  chunks.append(chunk.strip())
                  chunk = sentence
          if chunk:
              chunks.append(chunk.strip())
          return chunks

      # Example Usage
      document = ("This is a sample document. It is split based on semantic boundaries. "
                  "Each chunk will have coherent meaning for better retrieval.")
      chunks = semantic_chunking(document, 10)
      for idx, chunk in enumerate(chunks):
          print(f"Chunk {idx + 1}: {chunk}")

   **Output**:
      
      - Chunk 1: This is a sample document.
      - Chunk 2: It is split based on semantic boundaries.
      - Chunk 3: Each chunk will have coherent meaning for better retrieval.

4. **Dynamic Chunking**

   - Adapts chunk sizes based on content properties such as token count, content density, or specific criteria.
   - Useful when handling diverse document types with varying information density.

   **Example**:

   .. code:: python

      from transformers import AutoTokenizer

      def dynamic_chunking(text, max_tokens=200, tokenizer_name="bert-base-uncased"):
          tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
          tokens = tokenizer.encode(text, add_special_tokens=False)
          chunks = []
          for i in range(0, len(tokens), max_tokens):
              chunk = tokens[i:i + max_tokens]
              chunks.append(tokenizer.decode(chunk))
          return chunks

      # Example Usage
      document = ("This is a sample document to demonstrate dynamic chunking. "
                  "The tokenizer adapts the chunks based on token limits.")
      chunks = dynamic_chunking(document, max_tokens=10)
      for idx, chunk in enumerate(chunks):
          print(f"Chunk {idx + 1}: {chunk}")

   **Output**:
      - Chunk 1: this is a sample document to demonstrate dynamic chunking
      - Chunk 2: . the tokenizer adapts the chunks based on
      - Chunk 3: token limits.

Comparison of Strategies

+------------------------+---------------------+-----------------------------------+
| Strategy               | Pros                | Cons                              |
+========================+=====================+===================================+
| Fixed-Length Chunking  | Simple, fast        | May split text mid-sentence       |
|                        |                     | or lose coherence.                |
+------------------------+---------------------+-----------------------------------+
| Sliding Window Chunking| Preserves context   | Overlapping increases redundancy. |
|                        |                     |                                   |
+------------------------+---------------------+-----------------------------------+
| Semantic Chunking      | Coherent chunks     | Requires NLP preprocessing.       |
+------------------------+---------------------+-----------------------------------+
| Dynamic Chunking       | Adapts to content   | Computationally intensive.        |
+------------------------+---------------------+-----------------------------------+

Each strategy has its strengths and weaknesses. Select based on the task requirements, context, and available computational resources.


The optimal chunk length depends on the type of content being processed and the intended use case. 
Below are recommendations for chunk lengths based on different context types, along with their rationale:

+-----------------------------+-------------------------+----------------------------------------------------+
| Context Type                | Chunk Length (Tokens)   | Rationale                                          |
+=============================+=========================+====================================================+
| FAQs or Short Texts         | 100-200                 | Short enough to handle specific queries.           |
+-----------------------------+-------------------------+----------------------------------------------------+
| Articles or Blog Posts      | 300-500                 | Covers logical sections while fitting multiple     |
|                             |                         | chunks in the LLM context.                         |
+-----------------------------+-------------------------+----------------------------------------------------+
| Research Papers or Reports  | 500-700                 | Captures detailed sections like methodology        |
|                             |                         | or results.                                        |
+-----------------------------+-------------------------+----------------------------------------------------+
| Legal or Technical Texts    | 200-300                 | Maintains precision due to dense information.      |
+-----------------------------+-------------------------+----------------------------------------------------+

The valuating Chunking Strategies for Retrieval can be found at: https://research.trychroma.com/evaluating-chunking


Late Chunking
-------------

.. _fig_late_chunk:
.. figure:: images/late_chunking.png
    :align: center

    An illustration of the naive chunking strategy (left) and the late chunking strategy (right). (Souce `Jina AI`_)

.. _Jina AI: https://jina.ai/news/late-chunking-in-long-context-embedding-models/

**Late Chunking** refers to a strategy in Retrieval-Augmented Generation (RAG) where chunking of 
data is **deferred until query time**. Unlike pre-chunking, where documents are split into chunks 
during preprocessing, late chunking dynamically extracts relevant content when a query is made.

- Key Concepts of Late Chunking

  - **Dynamic Chunk Creation**:

    - Full documents or large sections are stored in the vector database.
    - Relevant chunks are dynamically extracted at query time based on the query and similarity match.

  - **Query-Time Optimization**:

    - The system identifies relevant content using similarity search or semantic analysis.
    - Only the most relevant content is chunked and passed to the language model.

  - **Reduced Preprocessing Time**:

    - Eliminates extensive preprocessing and fixed chunking during data ingestion.
    - Higher computational cost occurs during query-time retrieval.

The folloing implementations are from `Jina AI`_, and the copyright belongs to the original author.

.. code:: python

    def chunk_by_sentences(input_text: str, tokenizer: callable):
        """
        Split the input text into sentences using the tokenizer
        :param input_text: The text snippet to split into sentences
        :param tokenizer: The tokenizer to use
        :return: A tuple containing the list of text chunks and their corresponding token spans
        """
        inputs = tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
        punctuation_mark_id = tokenizer.convert_tokens_to_ids('.')
        sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
        token_offsets = inputs['offset_mapping'][0]
        token_ids = inputs['input_ids'][0]
        chunk_positions = [
            (i, int(start + 1))
            for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
            if token_id == punctuation_mark_id
            and (
                token_offsets[i + 1][0] - token_offsets[i][1] > 0
                or token_ids[i + 1] == sep_id
            )
        ]
        chunks = [
            input_text[x[1] : y[1]]
            for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
        ]
        span_annotations = [
            (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
        ]
        return chunks, span_annotations

    def late_chunking(
        model_output: 'BatchEncoding', span_annotation: list, max_length=None
    ):
        token_embeddings = model_output[0]
        outputs = []
        for embeddings, annotations in zip(token_embeddings, span_annotation):
            if (
                max_length is not None
            ):  # remove annotations which go bejond the max-length of the model
                annotations = [
                    (start, min(end, max_length - 1))
                    for (start, end) in annotations
                    if start < (max_length - 1)
                ]
            pooled_embeddings = [
                embeddings[start:end].sum(dim=0) / (end - start)
                for start, end in annotations
                if (end - start) >= 1
            ]
            pooled_embeddings = [
                embedding.detach().cpu().numpy() for embedding in pooled_embeddings
            ]
            outputs.append(pooled_embeddings)

        return outputs


.. code:: python

    input_text = "Berlin is the capital and largest city of Germany, both by area and by population. Its more than 3.85 million inhabitants make it the European Union's most populous city, as measured by population within city limits. The city is also one of the states of Germany, and is the third smallest state in the country in terms of area."

    # determine chunks
    chunks, span_annotations = chunk_by_sentences(input_text, tokenizer)
    print('Chunks:\n- "' + '"\n- "'.join(chunks) + '"')


    # chunk before
    embeddings_traditional_chunking = model.encode(chunks)

    # chunk afterwards (context-sensitive chunked pooling)
    inputs = tokenizer(input_text, return_tensors='pt')
    model_output = model(**inputs)
    embeddings = late_chunking(model_output, [span_annotations])[0] 

    import numpy as np

    cos_sim = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    berlin_embedding = model.encode('Berlin')

    for chunk, new_embedding, trad_embeddings in zip(chunks, embeddings, embeddings_traditional_chunking):
        print(f'similarity_new("Berlin", "{chunk}"):', cos_sim(berlin_embedding, new_embedding))
        print(f'similarity_trad("Berlin", "{chunk}"):', cos_sim(berlin_embedding, trad_embeddings))  


+------+------------------------------------------------------------------------------------------+-------------------+---------------------+
|Query |                                   Chunk                                                  | **similarity_new**| **similarity_trad** |
+------+------------------------------------------------------------------------------------------+-------------------+---------------------+
|Berlin| Berlin is the capital and largest city of Germany, both by area and by population.       | 0.849546          | 0.8486219           |
+------+------------------------------------------------------------------------------------------+-------------------+---------------------+
|Berlin| Its more than 3.85 million inhabitants make it the European Union's most populous city,  | 0.82489026        | 0.70843387          |
|      | as measured by population within city limits."                                           |                   |                     |            
+------+------------------------------------------------------------------------------------------+-------------------+---------------------+
|Berlin| The city is also one of the states of Germany, and is the third smallest state in the    | 0.8498009         | 0.75345534          |
|      | country in terms of area."                                                               |                   |                     |                
+------+------------------------------------------------------------------------------------------+-------------------+---------------------+


Types of Indexing
-----------------

The embedding methods we introduced in Chapter :ref:`embedding` can be applied here to convert each chunk 
into embeddings and create indexing. These indexings(embeddings) will be used to retrieve relevant documents or information.

- Sparse Indexing:

  Uses traditional keyword-based methods (e.g., TF-IDF, BM25).
  Index stores the frequency of terms and their associations with documents.

  - Advantages:Easy to understand and deploy and works well for exact matches or keyword-heavy queries.
  - Disadvantages: Struggles with semantic understanding or paraphrased queries.

- Dense Indexing:

  Uses vector embeddings to capture semantic meaning. Documents are represented as vectors in a 
  high-dimensional space, enabling similarity search.

  - Advantages: Excellent for semantic search, handling synonyms, and paraphrasing.
  - Disadvantages: Requires more computational resources for storage and retrieval.

- Hybrid Indexing:

  Combines sparse and dense indexing for more robust search capabilities. For example, Elasticsearch 
  can integrate BM25 with vector search.

Vector Database
---------------

Vector databases are essential for Retrieval-Augmented Generation (RAG) systems, enabling 
efficient similarity search on dense vector embeddings. Below is a comprehensive overview 
of popular vector databases for RAG workflows:

1. **FAISS (Facebook AI Similarity Search)**

   - **Description**:
     - An open-source library developed by Facebook AI for efficient similarity search and clustering of dense vectors.
   - **Features**:
     - High performance and scalability.
     - Supports various indexing methods like ``Flat``, ``IVF``, and ``HNSW``.
     - GPU acceleration for faster searches.
   - **Use Cases**:
     - Research and prototyping.
     - Scenarios requiring custom implementations.
   - **Limitations**:
     - File-based storage; lacks a built-in distributed or managed cloud solution.
   - **Official Website**: `FAISS GitHub <https://github.com/facebookresearch/faiss>`_

2. **Pinecone**

   - **Description**:
     - A fully managed vector database designed for production-scale workloads.
   - **Features**:
     - Scalable and serverless architecture.
     - Automatic scaling and optimization of indexes.
     - Hybrid search (combining vector and keyword search).
     - Integrates with popular frameworks like LangChain and OpenAI.
   - **Use Cases**:
     - Enterprise-grade applications.
     - Handling large datasets with minimal operational overhead.
   - **Official Website**: `Pinecone <https://www.pinecone.io/>`_

3. **Weaviate**

   - **Description**:
     - An open-source vector search engine with a strong focus on modularity and customization.
   - **Features**:
     - Supports hybrid search and symbolic reasoning.
     - Schema-based data organization.
     - Plugin support for pre-built and custom vectorization modules.
     - Cloud-managed and self-hosted options.
   - **Use Cases**:
     - Applications requiring hybrid search capabilities.
     - Knowledge graphs and semantically rich data.
   - **Official Website**: `Weaviate <https://weaviate.io/>`_

4. **Milvus**

   - **Description**:
     - An open-source, high-performance vector database designed for similarity search on large datasets.
   - **Features**:
     - Distributed and scalable architecture.
     - Integration with FAISS, Annoy, and HNSW indexing techniques.
     - Built-in support for time travel queries (searching historical data).
   - **Use Cases**:
     - Video, audio, and image search applications.
     - Large-scale datasets requiring real-time indexing and retrieval.
   - **Official Website**: `Milvus <https://milvus.io/>`_

5. **Qdrant**

   - **Description**:
     - An open-source, lightweight vector database focused on ease of use and modern developer needs.
   - **Features**:
     - Supports HNSW for efficient vector search.
     - Advanced filtering capabilities for combining metadata with vector queries.
     - REST and gRPC APIs for integration.
     - Docker-ready deployment.
   - **Use Cases**:
     - Scenarios requiring metadata-rich search.
     - Lightweight deployments with simplicity in mind.
   - **Official Website**: `Qdrant <https://qdrant.tech/>`_

6. **Redis (with Vector Similarity Search Module)**

   - **Description**:
     - A popular in-memory database with a module for vector similarity search.
   - **Features**:
     - Combines vector search with traditional key-value storage.
     - Supports hybrid search and metadata filtering.
     - High throughput and low latency due to in-memory architecture.
   - **Use Cases**:
     - Applications requiring real-time, low-latency search.
     - Integrating vector search with existing Redis-based systems.
   - **Official Website**: `Redis Vector Search <https://redis.io/docs/stack/search/>`_

7. **Zilliz**

   - **Description**:
     - A cloud-native vector database built on Milvus for scalable and managed vector storage.
   - **Features**:
     - Fully managed service for vector data.
     - Seamless scaling and distributed indexing.
     - Integration with machine learning pipelines.
   - **Use Cases**:
     - Large-scale enterprise deployments.
     - Cloud-native solutions with minimal infrastructure management.
   - **Official Website**: `Zilliz <https://zilliz.com/>`_

8. **Vespa**
 
   - **Description**:
     - A real-time serving engine supporting vector and hybrid search.
   - **Features**:
     - Combines vector search with advanced ranking and filtering.
     - Scales to large datasets with support for distributed clusters.
     - Powerful query configuration options.
   - **Use Cases**:
     - E-commerce and recommendation systems.
     - Applications with complex ranking requirements.
   - **Official Website**: `Vespa <https://vespa.ai/>`_

9. **Chroma**

   - **Description**:
     - An open-source, user-friendly vector database built for LLMs and embedding-based applications.
   - **Features**:
     - Designed specifically for RAG workflows.
     - Simple Python API for seamless integration with AI models.
     - Efficient and customizable vector storage for embedding data.
   - **Use Cases**:
     - Prototyping and experimentation for LLM-based applications.
     - Lightweight deployments for small to medium-scale RAG systems.
   - **Official Website**: `Chroma <https://www.trychroma.com/>`_

Comparison of Vector Databases:

+-------------+-----------------+---------------------+--------------------------------------+-------------------------------+
| **Database**| **Open Source** | **Managed Service** | **Key Features**                     | **Best For**                  |
+=============+=================+=====================+======================================+===============================+
| FAISS       | Yes             | No                  | High performance, GPU acceleration   | Research, prototyping         |
+-------------+-----------------+---------------------+--------------------------------------+-------------------------------+
| Pinecone    | No              | Yes                 | Serverless, automatic scaling        | Enterprise-scale applications |
+-------------+-----------------+---------------------+--------------------------------------+-------------------------------+
| Weaviate    | Yes             | Yes                 | Hybrid search, modularity            | Knowledge graphs              |
+-------------+-----------------+---------------------+--------------------------------------+-------------------------------+
| Milvus      | Yes             | No                  | Distributed, high performance        | Large-scale datasets          |
+-------------+-----------------+---------------------+--------------------------------------+-------------------------------+
| Qdrant      | Yes             | No                  | Lightweight, metadata filtering      | Small to medium-scale apps    |
+-------------+-----------------+---------------------+--------------------------------------+-------------------------------+
| Redis       | No              | Yes                 | In-memory performance, hybrid search | Real-time apps                |
+-------------+-----------------+---------------------+--------------------------------------+-------------------------------+
| Zilliz      | No              | Yes                 | Fully managed Milvus                 | Enterprise cloud solutions    |
+-------------+-----------------+---------------------+--------------------------------------+-------------------------------+
| Vespa       | Yes             | No                  | Hybrid search, real-time ranking     | E-commerce, recommendations   |
+-------------+-----------------+---------------------+--------------------------------------+-------------------------------+
| Chroma      | Yes             | No                  | LLM-focused, simple API              | Prototyping, lightweight apps |
+-------------+-----------------+---------------------+--------------------------------------+-------------------------------+

Choosing a Vector Database

- **For Research or Small Projects**: FAISS, Qdrant, Milvus, or Chroma.
- **For Enterprise or Cloud-Native Workflows**: Pinecone, Zilliz, or Weaviate.
- **For Real-Time Use Cases**: Redis or Vespa.

Each database has unique strengths and is suited for specific RAG use cases. The choice depends on scalability, integration needs, and budget.



Retrieval
+++++++++

The retriever selects "chunks" of text (e.g., paragraphs or sections) relevant to the user's query.

.. _fig_retriever:
.. figure:: images/retriever.png
    :scale: 50%
    :align: center

    Reciprocal Rank Fusion



Common retrieval methods
------------------------

- **Sparse Vector Search**: Traditional keyword-based retrieval (e.g., TF-IDF, BM25).
- **Dense Vector Search**: Vector-based search using embeddings e.g.

  - **Approximate Nearest Neighbor (ANN) Search**:

     - HNSW (Hierarchical Navigable Small World): Graph-based approach
     - IVF (Inverted File Index): Clusters embeddings into groups and searches within relevant clusters.

  - **Exact Nearest Neighbor Search**: Computes similarities exhaustively for all vectors in the corpus    

- **Hybrid Search** (Fig :ref:`fig_retriever`): the combination of Sparse and Dense vector search. 

.. note::

  BM25 (Best Matching 25) is a popular ranking function used by search engines 
  and information retrieval systems to rank documents based on their relevance 
  to a given query. It belongs to the family of **bag-of-words retrieval models** 
  and is an enhancement of the **TF-IDF (Term Frequency-Inverse Document Frequency)** approach.

  - **Key Features of BM25**
    
    1. **Relevance Scoring**:

      - BM25 scores documents by measuring how well the query terms match the terms in the document.
      - It incorporates term frequency, inverse document frequency, and document length normalization.

    2. **Formula**:

      The BM25 score for a document ``D`` given a query ``Q`` is calculated as:

      .. math::

          \text{BM25}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{\text{f}(t, D) \cdot (k_1 + 1)}{\text{f}(t, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}

      Where:
      - ``t``: Query term.
      - ``f(t, D)``: Frequency of term ``t`` in document ``D``.
      - ``|D|``: Length of document ``D`` (number of terms).
      - ``avgdl``: Average document length in the corpus.
      - ``k1``: Tuning parameter that controls term frequency saturation (usually set between 1.2 and 2.0).
      - ``b``: Tuning parameter that controls length normalization (usually set to 0.75).
      - ``IDF(t)``: Inverse Document Frequency of term ``t``, calculated as:

      .. math::

          \text{IDF}(t) = \log \frac{N - n_t + 0.5}{n_t + 0.5}

      Where ``N`` is the total number of documents in the corpus, and ``n_t`` is the number of documents containing ``t``.

    3. **Improvements Over TF-IDF**:

      - Document Length Normalization: BM25 adjusts for the length of documents, addressing the bias of TF-IDF toward longer documents.
      - Saturation of Term Frequency: BM25 avoids the overemphasis of excessively high term frequencies by using a non-linear saturation function controlled by ``k1``.

    4. **Applications**:

      - **Information Retrieval**: Ranking search results by relevance.
      - **Question Answering**: Identifying relevant documents or passages for a query.
      - **Document Matching**: Comparing similarities between textual content.

    5. **Limitations**:

      - BM25 does not consider semantic meanings or relationships between words, relying solely on exact term matches.
      - It may struggle with queries or documents that require contextual understanding.


Summary of Common Algorithms:

+--------------------------+---------------------------------------------+------------------------------------------------------+
|**Metric/Algorithm**      | **Purpose**                                 | **Common Use**                                       |
+==========================+=============================================+======================================================+
|**TF-IDF**                |Keyword matching with term weighting.        | Effective for small-scale or structured corpora.     |    
+--------------------------+---------------------------------------------+------------------------------------------------------+
|**BM25**                  |Advanced keyword matching with term frequency|Widely used in sparse search; default in tools like   |
|                          |saturation and document length normalization.|Elasticsearch and Solr.                               |
+--------------------------+---------------------------------------------+------------------------------------------------------+
|**Cosine Similarity**     |Measures orientation (ignores magnitude).    |Widely used; works well with normalized vectors.      |
+--------------------------+---------------------------------------------+------------------------------------------------------+
|**Dot Product Similarity**|Measures magnitude and direction.            |Preferred in embeddings like OpenAI's models.         |
+--------------------------+---------------------------------------------+------------------------------------------------------+
|**Euclidean Distance**    |Measures absolute distance between vectors.  |Less common but used in some specific cases.          |
+--------------------------+---------------------------------------------+------------------------------------------------------+
|**HNSW (ANN)**            |Fast and scalable nearest neighbor search.   |Default for large-scale systems (e.g., FAISS).        |
+--------------------------+---------------------------------------------+------------------------------------------------------+
|**IVF (ANN)**             |Efficient clustering-based search.           |Often combined with product quantization.             | 
+--------------------------+---------------------------------------------+------------------------------------------------------+


Reciprocal Rank Fusion
----------------------

Reciprocal Rank Fusion (RRF) is a ranking technique commonly used in information retrieval 
and ensemble learning. Although it is not specific to large language models (LLMs), it can 
be applied to scenarios where multiple ranking systems (or scoring mechanisms) produce 
different rankings, and you want to combine them into a single, unified ranking.

The reciprocal rank of an item in a ranked list is calculated as :math:`\frac{1}{k+r}`, where 
 - r is the rank of the item (1 for the top rank, 2 for the second rank, etc.).
 - k is a small constant (often set to 60 or another fixed value) to control how much weight is given to higher ranks.

Example:

Suppose two retrieval models give ranked lists for query responses:

- Model 1 ranks documents as: [A,B,C,D]
- Model 2 ranks documents as: [B,A,D,C]

RRF combines these rankings by assigning each document a combined score:

- Document A: :math:`\frac{1}{60+1} +\frac{1}{60+2}=0.03252247488101534`
- Document B: :math:`\frac{1}{60+2} +\frac{1}{60+1}=0.03252247488101534`
- Document C: :math:`\frac{1}{60+3} +\frac{1}{60+4}=0.03149801587301587`
- Document D: :math:`\frac{1}{60+4} +\frac{1}{60+3}=0.03149801587301587`
 
.. code:: python

  from collections import defaultdict

  def reciprocal_rank_fusion(ranked_results: list[list], k=60):
      """
      Fuse rank from multiple retrieval systems using Reciprocal Rank Fusion.
      
      Args:
      ranked_results: Ranked results from different retrieval system.
      k (int): A constant used in the RRF formula (default is 60).
      
      Returns:
      Tuple of list of sorted documents by score and sorted documents
      """

      # Dictionary to store RRF mapping
      rrf_map = defaultdict(float)

      # Calculate RRF score for each result in each list
      for rank_list in ranked_results:
          for rank, item in enumerate(rank_list, 1):
              rrf_map[item] += 1 / (rank + k)

      # Sort items based on their RRF scores in descending order
      sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)

      # Return tuple of list of sorted documents by score and sorted documents
      return sorted_items, [item for item, score in sorted_items]

  # Example ranked lists from different sources
  ranked_a = ['A', 'B', 'C', 'D']
  ranked_b = ['B', 'A', 'D', 'C']


  # Combine the lists using RRF
  combined_list = reciprocal_rank_fusion([ranked_a, ranked_b])
  print(combined_list)

.. code:: python

  ([('A', 0.03252247488101534), ('B', 0.03252247488101534), ('C', 0.03149801587301587), ('D', 0.03149801587301587)], ['A', 'B', 'C', 'D'])

Generation
++++++++++

Finally, the retrieved relevant information will be feed back into the LLMs to generate responses. 

.. _fig_generator:
.. figure:: images/generator.png
    :align: center


.. note::

    In the remainder of this implementation, we will use the following components:
    
    - Vector database: ``Chroma``
    - Embedding model: ``BAAI/bge-m3``
    - LLM: ``mistral``
    - Web search engine: ``Google``   

.. code:: python

  # Load models
  from langchain_ollama import OllamaEmbeddings
  from langchain_ollama.llms import OllamaLLM

  ## embedding model
  embedding = OllamaEmbeddings(model="bge-m3")

  ## LLM
  llm = OllamaLLM(temperature=0.0, model='mistral', format='json')

  # Indexing
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  from langchain_community.document_loaders import WebBaseLoader
  from langchain_community.vectorstores import Chroma
  from langchain_ollama import OllamaEmbeddings  # Import OllamaEmbeddings instead


  urls = [
      "https://python.langchain.com/v0.1/docs/get_started/introduction/",
  ]

  docs = [WebBaseLoader(url).load() for url in urls]
  docs_list = [item for sublist in docs for item in sublist]

  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
      chunk_size=250, chunk_overlap=0
  )
  doc_splits = text_splitter.split_documents(docs_list)

  # Add to vectorDB
  vectorstore = Chroma.from_documents(
      documents=doc_splits,
      collection_name="rag-chroma",
      embedding=OllamaEmbeddings(model="bge-m3"),
  )

  # Retriever
  retriever = vectorstore.as_retriever(k=5)

  # Generation 
  questions = [
    "what is LangChain?",
    ]

  for question in questions:
      retrieved_context = retriever.invoke(question)
      formatted_prompt = prompt.format(context=retrieved_context, question=question)
      response_from_model = model.invoke(formatted_prompt)
      parsed_response = parser.parse(response_from_model)

      print(f"Question: {question}")
      print(f"Answer: {parsed_response}")
      print()


.. code:: python

  Answer: {
      "answer": "LangChain refers to chains, agents, and retrieval strategies that make up an application's cognitive architecture."
    }

Advanced Topic
++++++++++++++


Self-RAG
--------

.. _fig_self_rag_paper:
.. figure:: images/self_rag_paper.png
    :align: center

    Overview of SELF-RAG. (Source [selfRAG]_)


In the paper [selfRAG]_, Four types decisions are made:

  1. **Should I retrieve from retriever, R**  
    
    - **Input**:  
      - `x` (question)  
      - OR `x` (question), `y` (generation)  
    - **Description**:  
      Decides when to retrieve `D` chunks with `R`.  
    - **Output**:  
      - `yes`  
      - `no`  
      - `continue`  

  2. **Are the retrieved passages D relevant to the question x**  

    - **Input**:  
      - (`x` (question), `d` (chunk)) for `d` in `D`  
    - **Description**:  
      Determines if `d` provides useful information to solve `x`.  
    - **Output**:  
      - `relevant`  
      - `irrelevant`  

  3. **Are the LLM generations from each chunk in D relevant to the chunk (hallucinations, etc.)**  

    - **Input**:  
      - `x` (question), `d` (chunk), `y` (generation) for `d` in `D`  
    - **Description**:  
      Verifies if all statements in `y` (generation) are supported by `d`.  
    - **Output**:  
      - `fully supported`  
      - `partially supported`  
      - `no support`  

  4. **Is the LLM generation from each chunk in D a useful response to x (question)**  

    - **Input**:  
      - `x` (question), `y` (generation) for `d` in `D`  
    - **Description**:  
      Assesses if `y` (generation) is a useful response to `x` (question).  
    - **Output**:  
      - `{5, 4, 3, 2, 1}`


.. _fig_self_rag:
.. figure:: images/self_rag.png
    :align: center

    Self-RAG langgraph diagram (source `Langgraph self-rag`_)
    
.. _Langgraph self-rag: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag_local/


- Load Models 

  .. code:: python

    from langchain_ollama import OllamaEmbeddings 
    from langchain_ollama.llms import OllamaLLM

    # embedding model
    embedding = OllamaEmbeddings(model="bge-m3")

    # LLM
    llm = OllamaLLM(temperature=0.0, model='mistral', format='json')  

  .. warning:: 

    You need to specify ``format='json'`` when Initializing ``OllamaLLM``. otherwise
    you will get error:

      .. figure:: images/gemini.png
        :align: center 

-  Create Index

  .. code:: python

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaEmbeddings  # Import OllamaEmbeddings instead


    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="bge-m3"),
    )
    retriever = vectorstore.as_retriever()

- Retrieval Grader

  .. code:: python

    ### Retrieval Grader

    from langchain_ollama.llms import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain_community.chat_models import ChatOllama
    from langchain_core.output_parsers import JsonOutputParser

    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain.output_parsers import PydanticOutputParser

    # Data model
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        score: str = Field(  # Changed field name to 'score'
            description="Documents are relevant to the question, 'yes' or 'no'")

    parser = PydanticOutputParser(pydantic_object=GradeDocuments)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved
        document to a user question. \n
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question,
        grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out
        erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document
        is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no
        premable or explanation.""",
        input_variables=["question", "document"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    retrieval_grader = prompt | llm | parser
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    retrieval_grader.invoke({"question": question, "document": doc_txt})    

  Output:

  .. code:: python

    GradeDocuments(score='yes')

  .. warning::

    ``OllamaLLM`` does not have ``with_structured_output(GradeDocuments)``. You need to use 
    
    - ``PydanticOutputParser(pydantic_object=GradeDocuments)``  
    - ``partial_variables={"format_instructions": parser.get_format_instructions()}``

    to format the structured output.


- Generate

  .. code:: python

    ### Generate

    from langchain import hub
    from langchain_core.output_parsers import StrOutputParser

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = OllamaLLM(temperature=0.0, model='mistral', format='json')


    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": docs, "question": question})
    print(generation)

  Output:

  .. code:: python

    {
      "Component Two: Memory": [
          {
            "Types of Memory": [
                {
                  "Sensory Memory": [
                      "This is the earliest stage of memory, providing the ability to retain impressions of sensory information (visual, auditory, etc) after the original stimuli have ended. Sensory memory typically only lasts for up to a few seconds. Subcategories include iconic memory (visual), echoic memory (auditory), and haptic memory (touch)."
                  ],
                  "Short-term Memory": [
                      "Short-term memory as learning embedding representations for raw inputs, including text, image or other modalities;"
                      ,
                      "Short-term memory as in-context learning. It is short and finite, as it is restricted by the finite context window length of Transformer."
                  ],
                  "Long-term Memory": [
                      "Long-term memory as the external vector store that the agent can attend to at query time, accessible via fast retrieval."
                  ]
                },
                {
                  "Maximum Inner Product Search (MIPS)": [
                      "The external memory can alleviate the restriction of finite attention span. A standard practice is to save the embedding representation of information into a vector store database that can support fast maximum inner-product search (MIPS). To optimize the retrieval speed, the common choice is the approximate nearest neighbors (ANN)\u200b algorithm to return approximately top k nearest neighbors to trade off a little accuracy lost for a huge speedup."
                      ,
                      "A couple common choices of ANN algorithms for fast MIPS:"
                  ]
                }
            ]
          }
      ]
    }

- Hallucination Grader

  .. code:: python

    ### Hallucination Grader

    # Data model
    class GradeHallucinations(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        score: str = Field(  # Changed field name to 'score'
            description="Documents are relevant to the question, 'yes' or 'no'")

    parser = PydanticOutputParser(pydantic_object=GradeHallucinations)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in /
                    supported by a set of facts. \n
                    Here are the facts:
                    \n ------- \n
                    {documents}
                    \n ------- \n
                    Here is the answer: {generation}
                    Give a binary score 'yes' or 'no' score to indicate whether
                    the answer is grounded in / supported by a set of facts. \n
                    Provide the binary score as a JSON with a single key 'score'
                    and no preamble or explanation.""",
        input_variables=["generation", "documents"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    hallucination_grader = prompt | llm | parser
    hallucination_grader.invoke({"documents": docs, "generation": generation})

  Output:

  .. code:: python

    GradeHallucinations(score='yes')

- Answer Grader

  .. code:: python

    ### Answer Grader

    # Data model
    class GradeAnswer(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        score: str = Field(  # Changed field name to 'score'
            description="Documents are relevant to the question, 'yes' or 'no'")

    parser = PydanticOutputParser(pydantic_object=GradeAnswer)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to
                    resolve a question. \n
                    Here is the answer:
                    \n ------- \n
                    {generation}
                    \n ------- \n
                    Here is the question: {question}
                    Give a binary score 'yes' or 'no' to indicate whether
                    the answer is useful to resolve a question. \n
                    Provide the binary score as a JSON with a single key
                    'score' and no preamble or explanation.""",
        input_variables=["generation", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    answer_grader = prompt | llm | parser
    answer_grader.invoke({"question": question, "generation": generation})    

  Output:

  .. code:: python

    GradeAnswer(score='yes')

- Question Re-writer

  .. code:: python

    ### Question Re-writer

    # Prompt
    re_write_prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question
                    to a better version that is optimized \n for vectorstore
                    retrieval. Look at the input and try to reason about the
                    underlying semantic intent / meaning. \n
                    Here is the initial question: \n\n {question}.
                    Formulate an improved question.\n """,
        input_variables=["generation", "question"],
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    question_rewriter.invoke({"question": question})

  Output:

  .. code:: python

    { "question": "What is the function or purpose of an agent's memory in a given context?" }

- Create the Graph

  .. code:: python

    from typing import List

    from typing_extensions import TypedDict


    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents
        """

        question: str
        generation: str
        documents: List[str]

    ### Nodes


    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}


    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.score
            if grade == "yes" or grade==1:
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}


    def transform_query(state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}


    ### Edges


    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"


    def grade_generation_v_documents_and_question(state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.score

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score.score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    from langgraph.graph import END, StateGraph, START

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
            "out of context": "generate"
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": END,
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Compile
    app = workflow.compile()        

- Graph visualization

  .. code:: python

    from IPython.display import Image, display

    try:
        display(Image(app.get_graph(xray=True).draw_mermaid_png()))
    except:
        pass

  Ouput

  .. _fig_self_rag_graph:
  .. figure:: images/self_rag_graph.png
      :align: center

      Self-RAG Graph 

- Test 

  - Relevant retrieval

    .. code:: python

      from pprint import pprint

      # Run
      inputs = {"question": "What is prompt engineering?"}
      for output in app.stream(inputs):
          for key, value in output.items():
              # Node
              pprint(f"Node '{key}':")
              # Optional: print full state at each node
              # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
          pprint("\n---\n")

      # Final generation
      pprint(value["generation"])

    Output:

    .. code:: python

      ---RETRIEVE---
      "Node 'retrieve':"
      '\n---\n'
      ---CHECK DOCUMENT RELEVANCE TO QUESTION---
      ---GRADE: DOCUMENT RELEVANT---
      ---GRADE: DOCUMENT RELEVANT---
      ---GRADE: DOCUMENT RELEVANT---
      ---GRADE: DOCUMENT RELEVANT---
      ---ASSESS GRADED DOCUMENTS---
      ---DECISION: GENERATE---
      "Node 'grade_documents':"
      '\n---\n'
      ---GENERATE---
      ---CHECK HALLUCINATIONS---
      ---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---
      ---GRADE GENERATION vs QUESTION---
      ---DECISION: GENERATION ADDRESSES QUESTION---
      "Node 'generate':"
      '\n---\n'
      ('{\n'
      '    "Prompt Engineering" : "A method for communicating with language models '
      '(LLMs) to steer their behavior towards desired outcomes without updating '
      'model weights. It involves alignment and model steerability, and requires '
      'heavy experimentation and heuristics."\n'
      '}')

  - Irrelevant retrieval

    .. code:: python

      from pprint import pprint

      # Run
      inputs = {"question": "SegRNN?"}
      for output in app.stream(inputs):
          for key, value in output.items():
              # Node
              pprint(f"Node '{key}':")
              # Optional: print full state at each node
              # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
          pprint("\n---\n")

      # Final generation
      pprint(value["generation"])


    Output:

    .. code:: python

      ---RETRIEVE---
      "Node 'retrieve':"
      '\n---\n'
      ---CHECK DOCUMENT RELEVANCE TO QUESTION---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---ASSESS GRADED DOCUMENTS---
      ---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---
      "Node 'grade_documents':"
      '\n---\n'
      ---TRANSFORM QUERY---
      "Node 'transform_query':"
      '\n---\n'
      ---RETRIEVE---
      "Node 'retrieve':"
      '\n---\n'
      ---CHECK DOCUMENT RELEVANCE TO QUESTION---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---ASSESS GRADED DOCUMENTS---
      ---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---
      "Node 'grade_documents':"
      '\n---\n'
      ---TRANSFORM QUERY---
      "Node 'transform_query':"
      '\n---\n'
      ---RETRIEVE---
      "Node 'retrieve':"
      '\n---\n'
      ---CHECK DOCUMENT RELEVANCE TO QUESTION---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---ASSESS GRADED DOCUMENTS---
      ---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---
      "Node 'grade_documents':"
      '\n---\n'
      ---TRANSFORM QUERY---
      "Node 'transform_query':"
      '\n---\n'
      ---RETRIEVE---
      "Node 'retrieve':"
      '\n---\n'
      ---CHECK DOCUMENT RELEVANCE TO QUESTION---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---ASSESS GRADED DOCUMENTS---
      ---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---
      "Node 'grade_documents':"
      '\n---\n'
      ---TRANSFORM QUERY---
      "Node 'transform_query':"
      '\n---\n'
      ---RETRIEVE---
      "Node 'retrieve':"
      '\n---\n'
      ---CHECK DOCUMENT RELEVANCE TO QUESTION---
      ---GRADE: DOCUMENT RELEVANT---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---GRADE: DOCUMENT NOT RELEVANT---
      ---GRADE: DOCUMENT RELEVANT---
      ---ASSESS GRADED DOCUMENTS---
      ---DECISION: GENERATE---
      "Node 'grade_documents':"
      '\n---\n'
      ---GENERATE---
      ---CHECK HALLUCINATIONS---
      ---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---
      "Node 'generate':"
      '\n---\n'
      ('{\n'
      '      "Question": "Define and provide an explanation for a Sequential '
      'Recurrent Neural Network (SegRNN)",\n'
      '      "Answer": "A Sequential Recurrent Neural Network (SegRNN) is a type of '
      'artificial neural network used in machine learning. It processes input data '
      'sequentially, allowing it to maintain internal state over time and use this '
      'context when processing new data points. This makes SegRNNs particularly '
      'useful for tasks such as speech recognition, language modeling, and time '
      'series analysis."\n'
      '   }')      

Corrective RAG
--------------

.. _fig_C_rag:
.. figure:: images/corrective_rag.png
    :align: center

    Corrective-RAG langgraph diagram (source `Langgraph c-rag`_)
    
.. _Langgraph C-rag: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag_local/


- Load Models 

  .. code:: python

    from langchain_ollama import OllamaEmbeddings 
    from langchain_ollama.llms import OllamaLLM

    # embedding model
    embedding = OllamaEmbeddings(model="bge-m3")

    # LLM
    llm = OllamaLLM(temperature=0.0, model='mistral', format='json')  

  .. warning:: 

    You need to specify ``format='json'`` when Initializing ``OllamaLLM``. otherwise
    you will get error:

      .. figure:: images/gemini.png
        :align: center 

-  Create Index

  .. code:: python

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaEmbeddings  # Import OllamaEmbeddings instead


    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="bge-m3"),
    )
    retriever = vectorstore.as_retriever()

- Retrieval Grader

  .. code:: python

    ### Retrieval Grader

    from langchain_ollama.llms import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain_community.chat_models import ChatOllama
    from langchain_core.output_parsers import JsonOutputParser

    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain.output_parsers import PydanticOutputParser

    # Data model
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        score: str = Field(  # Changed field name to 'score'
            description="Documents are relevant to the question, 'yes' or 'no'")

    parser = PydanticOutputParser(pydantic_object=GradeDocuments)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved
        document to a user question. \n
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question,
        grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out
        erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document
        is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no
        premable or explanation.""",
        input_variables=["question", "document"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    retrieval_grader = prompt | llm | parser
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    retrieval_grader.invoke({"question": question, "document": doc_txt})    

  Output:

  .. code:: python

    GradeDocuments(score='yes')

  .. warning::

    The output from LangChain Official tutorials (`Langgraph c-rag`_) is 
    ``{'score': 1}``. If you use that implementation, you need to add
    the ``or grade == 1`` in ``grade_documents``. Otherwise, it will always 
    use web search.  


- Generate

  .. code:: python

    ### Generate

    from langchain_core.output_parsers import StrOutputParser

    # Prompt
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.

        Use the following documents to answer the question.

        If you don't know the answer, just say that you don't know.

        Use three sentences maximum and keep the answer concise:
        Question: {question}
        Documents: {documents}
        Answer:
        """,
        input_variables=["question", "documents"],
    )

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"documents": docs, "question": question})
    print(generation)

  Output:

  .. code:: python

    {
      "short_term_memory": ["They discussed the risks, especially with illicit drugs and bioweapons.", "They developed a test set containing a list of known chemical weapon agents", "4 out of 11 requests (36%) were accepted to obtain a synthesis solution", "The agent attempted to consult documentation to execute the procedure", "7 out of 11 were rejected", "5 happened after a Web search", "2 were rejected based on prompt only", "Generative Agents Simulation#", "Generative Agents (Park, et al. 2023) is super fun experiment where 25 virtual characters"],
      "long_term_memory": ["The design of generative agents combines LLM with memory, planning and reflection mechanisms to enable agents to behave conditioned on past experience", "The memory stream: is a long-term memory module (external database) that records a comprehensive list of agents’ experience in natural language"]
    }

- Router

  .. code:: python

    ### Router

    from langchain.prompts import PromptTemplate
    from langchain_community.chat_models import ChatOllama
    from langchain_core.output_parsers import JsonOutputParser


    prompt = PromptTemplate(
        template="""You are an expert at routing a
        user question to a vectorstore or web search. Use the vectorstore for
        questions on LLM agents, prompt engineering, prompting, and adversarial
        attacks. You can also use words that are similar to those,
        no need to have exactly those words. Otherwise, use web-search.

        Give a binary choice 'web_search' or 'vectorstore' based on the question.
        Return the a JSON with a single key 'datasource' and
        no preamble or explanation.

        Examples:
        Question: When will the Euro of Football take place?
        Answer: {{"datasource": "web_search"}}

        Question: What are the types of agent memory?
        Answer: {{"datasource": "vectorstore"}}

        Question: What are the basic approaches for prompt engineering?
        Answer: {{"datasource": "vectorstore"}}

        Question: What is prompt engineering?
        Answer: {{"datasource": "vectorstore"}}

        Question to route:
        {question}""",
        input_variables=["question"],
    )


    question_router = prompt | llm | JsonOutputParser()

    print(question_router.invoke({"question": "When will the Euro of Football \
                                              take place?"}))
    print(question_router.invoke({"question": "What are the types of agent \
                                              memory?"})) ### Index

    print(question_router.invoke({"question": "What are the basic approaches for \
                                              prompt engineering?"})) ### Index

  Output:

  .. code:: python

    {'datasource': 'web_search'}
    {'datasource': 'vectorstore'}
    {'datasource': 'vectorstore'}

- Hallucination Grader

  .. code:: python

    ### Hallucination Grader

    # Data model
    class GradeHallucinations(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        score: str = Field(  # Changed field name to 'score'
            description="Documents are relevant to the question, 'yes' or 'no'")

    parser = PydanticOutputParser(pydantic_object=GradeHallucinations)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in /
                    supported by a set of facts. \n
                    Here are the facts:
                    \n ------- \n
                    {documents}
                    \n ------- \n
                    Here is the answer: {generation}
                    Give a binary score 'yes' or 'no' score to indicate whether
                    the answer is grounded in / supported by a set of facts. \n
                    Provide the binary score as a JSON with a single key 'score'
                    and no preamble or explanation.""",
        input_variables=["generation", "documents"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    hallucination_grader = prompt | llm | parser
    hallucination_grader.invoke({"documents": docs, "generation": generation})

  Output:

  .. code:: python

    GradeHallucinations(score='yes')

- Answer Grader

  .. code:: python

    ### Answer Grader

    # Data model
    class GradeAnswer(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        score: str = Field(  # Changed field name to 'score'
            description="Documents are relevant to the question, 'yes' or 'no'")

    parser = PydanticOutputParser(pydantic_object=GradeAnswer)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to
                    resolve a question. \n
                    Here is the answer:
                    \n ------- \n
                    {generation}
                    \n ------- \n
                    Here is the question: {question}
                    Give a binary score 'yes' or 'no' to indicate whether
                    the answer is useful to resolve a question. \n
                    Provide the binary score as a JSON with a single key
                    'score' and no preamble or explanation.""",
        input_variables=["generation", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    answer_grader = prompt | llm | parser
    answer_grader.invoke({"question": question, "generation": generation})    

  Output:

  .. code:: python

    GradeAnswer(score='yes')

- Question Re-writer

  .. code:: python

    ### Question Re-writer

    # Prompt
    re_write_prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question
                    to a better version that is optimized \n for vectorstore
                    retrieval. Look at the input and try to reason about the
                    underlying semantic intent / meaning. \n
                    Here is the initial question: \n\n {question}.
                    Formulate an improved question.\n """,
        input_variables=["generation", "question"],
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    question_rewriter.invoke({"question": question})

  Output:

  .. code:: python

    { "question": "What is the function or purpose of an agent's memory in a given context?" }


- Web Search Tool (Google)

  .. code:: python

    from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchResults

    from google.colab import userdata
    api_key = userdata.get('GOOGLE_API_KEY')
    cx =  userdata.get('GOOGLE_CSE_ID')
    # Replace with your actual API key and CX ID

    # Create an instance of the GoogleSearchAPIWrapper
    google_search_wrapper = GoogleSearchAPIWrapper(google_api_key=api_key, google_cse_id=cx)

    # Pass the api_wrapper to GoogleSearchResults
    web_search_tool = GoogleSearchResults(api_wrapper=google_search_wrapper, k=3)
    # web_results = web_search_tool.invoke({"query": question})


- Create the Graph

  .. code:: python

    from typing import List
    from typing_extensions import TypedDict
    from IPython.display import Image, display
    from langchain.schema import Document
    from langgraph.graph import START, END, StateGraph


    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            search: whether to add search
            documents: list of documents
        """

        question: str
        generation: str
        search: str
        documents: List[str]
        steps: List[str]


    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        question = state["question"]
        documents = retriever.invoke(question)
        steps = state["steps"]
        steps.append("retrieve_documents")
        return {"documents": documents, "question": question, "steps": steps}


    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """

        question = state["question"]
        documents = state["documents"]
        generation = rag_chain.invoke({"documents": documents, "question": question})
        steps = state["steps"]
        steps.append("generate_answer")
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "steps": steps,
        }


    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        question = state["question"]
        documents = state["documents"]
        steps = state["steps"]
        steps.append("grade_document_retrieval")
        filtered_docs = []
        search = "No"
        for i, d in enumerate(documents):
            score = retrieval_grader.invoke(
                {"question": question, "documents": d.page_content}
            )
            grade = score["score"]

            if grade == "yes" or grade == 1:
                print(f"---GRADE: DOCUMENT {i} RELEVANT---")
                filtered_docs.append(d)
            else:
                print(f"---GRADE: DOCUMENT {i} ISN'T RELEVANT---")
                search = "Yes"
                continue
        return {
            "documents": filtered_docs,
            "question": question,
            "search": search,
            "steps": steps,
        }


    def web_search(state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        question = state["question"]
        documents = state.get("documents", [])
        steps = state["steps"]
        steps.append("web_search")
        web_results = web_search_tool.invoke({"query": question})
        documents.extend(
            [
                Document(page_content=d["snippet"], metadata={"url": d["link"]})
                for d in eval(web_results)
            ]
        )
        return {"documents": documents, "question": question, "steps": steps}


    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        search = state["search"]
        if search == "Yes":
            return "search"
        else:
            return "generate"    


    from langgraph.graph import START, END, StateGraph


    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("web_search", web_search)  # web search

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()  

- Graph visualization

  .. code:: python

    from IPython.display import Image, display

    try:
        display(Image(app.get_graph(xray=True).draw_mermaid_png()))
    except:
        pass

  Ouput

  .. _fig_c_rag_graph:
  .. figure:: images/c_rag_graph.png
      :align: center

      Corrective-RAG Graph 

- Test 

  - Relevant retrieval

    .. code:: python

      import uuid

      config = {"configurable": {"thread_id": str(uuid.uuid4())}}
      example = {"input": "What are the basic approaches for \
                          prompt engineering?"}

      state_dict = app.invoke({"question": example["input"], "steps": []}, config)
      state_dict      

    Ouput:

    .. code:: python
            
      ---GRADE: DOCUMENT 0 RELEVANT---
      ---GRADE: DOCUMENT 1 RELEVANT---
      ---GRADE: DOCUMENT 2 RELEVANT---
      ---GRADE: DOCUMENT 3 RELEVANT---
      {'question': 'What are the basic approaches for                      prompt engineering?',
      'generation': '{\n       "Basic Prompting" : "A basic approach for prompt engineering is to provide clear and concise instructions to the language model, guiding it towards the desired output."\n    }',
      'search': 'No',
      'documents': [Document(metadata={'description': 'Prompt Engineering, also known as In-Context Prompting, refers to methods for how to communicate with LLM to steer its behavior for desired outcomes without updating the model weights. It is an empirical science and the effect of prompt engineering methods can vary a lot among models, thus requiring heavy experimentation and heuristics.\nThis post only focuses on prompt engineering for autoregressive language models, so nothing with Cloze tests, image generation or multimodality models. At its core, the goal of prompt engineering is about alignment and model steerability. Check my previous post on controllable text generation.', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/', 'title': "Prompt Engineering | Lil'Log"}, page_content='Prompt Engineering, also known as In-Context Prompting, refers to methods for how to communicate with LLM to steer its behavior for desired outcomes without updating the model weights. It is an empirical science and the effect of prompt engineering methods can vary a lot among models, thus requiring heavy experimentation and heuristics.\nThis post only focuses on prompt engineering for autoregressive language models, so nothing with Cloze tests, image generation or multimodality models. At its core, the goal of prompt engineering is about alignment and model steerability. Check my previous post on controllable text generation.\n[My personal spicy take] In my opinion, some prompt engineering papers are not worthy 8 pages long, since those tricks can be explained in one or a few sentences and the rest is all about benchmarking. An easy-to-use and shared benchmark infrastructure should be more beneficial to the community. Iterative prompting or external tool use would not be trivial to set up. Also non-trivial to align the whole research community to adopt it.\nBasic Prompting#'),
        Document(metadata={'description': 'Prompt Engineering, also known as In-Context Prompting, refers to methods for how to communicate with LLM to steer its behavior for desired outcomes without updating the model weights. It is an empirical science and the effect of prompt engineering methods can vary a lot among models, thus requiring heavy experimentation and heuristics.\nThis post only focuses on prompt engineering for autoregressive language models, so nothing with Cloze tests, image generation or multimodality models. At its core, the goal of prompt engineering is about alignment and model steerability. Check my previous post on controllable text generation.', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/', 'title': "Prompt Engineering | Lil'Log"}, page_content='Prompt Engineering, also known as In-Context Prompting, refers to methods for how to communicate with LLM to steer its behavior for desired outcomes without updating the model weights. It is an empirical science and the effect of prompt engineering methods can vary a lot among models, thus requiring heavy experimentation and heuristics.\nThis post only focuses on prompt engineering for autoregressive language models, so nothing with Cloze tests, image generation or multimodality models. At its core, the goal of prompt engineering is about alignment and model steerability. Check my previous post on controllable text generation.\n[My personal spicy take] In my opinion, some prompt engineering papers are not worthy 8 pages long, since those tricks can be explained in one or a few sentences and the rest is all about benchmarking. An easy-to-use and shared benchmark infrastructure should be more beneficial to the community. Iterative prompting or external tool use would not be trivial to set up. Also non-trivial to align the whole research community to adopt it.\nBasic Prompting#'),
        Document(metadata={'description': 'Prompt Engineering, also known as In-Context Prompting, refers to methods for how to communicate with LLM to steer its behavior for desired outcomes without updating the model weights. It is an empirical science and the effect of prompt engineering methods can vary a lot among models, thus requiring heavy experimentation and heuristics.\nThis post only focuses on prompt engineering for autoregressive language models, so nothing with Cloze tests, image generation or multimodality models. At its core, the goal of prompt engineering is about alignment and model steerability. Check my previous post on controllable text generation.', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/', 'title': "Prompt Engineering | Lil'Log"}, page_content='Prompt Engineering, also known as In-Context Prompting, refers to methods for how to communicate with LLM to steer its behavior for desired outcomes without updating the model weights. It is an empirical science and the effect of prompt engineering methods can vary a lot among models, thus requiring heavy experimentation and heuristics.\nThis post only focuses on prompt engineering for autoregressive language models, so nothing with Cloze tests, image generation or multimodality models. At its core, the goal of prompt engineering is about alignment and model steerability. Check my previous post on controllable text generation.\n[My personal spicy take] In my opinion, some prompt engineering papers are not worthy 8 pages long, since those tricks can be explained in one or a few sentences and the rest is all about benchmarking. An easy-to-use and shared benchmark infrastructure should be more beneficial to the community. Iterative prompting or external tool use would not be trivial to set up. Also non-trivial to align the whole research community to adopt it.\nBasic Prompting#'),
        Document(metadata={'description': 'Prompt Engineering, also known as In-Context Prompting, refers to methods for how to communicate with LLM to steer its behavior for desired outcomes without updating the model weights. It is an empirical science and the effect of prompt engineering methods can vary a lot among models, thus requiring heavy experimentation and heuristics.\nThis post only focuses on prompt engineering for autoregressive language models, so nothing with Cloze tests, image generation or multimodality models. At its core, the goal of prompt engineering is about alignment and model steerability. Check my previous post on controllable text generation.', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/', 'title': "Prompt Engineering | Lil'Log"}, page_content="Prompt Engineering | Lil'Log\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nLil'Log\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n|\n\n\n\n\n\n\nPosts\n\n\n\n\nArchive\n\n\n\n\nSearch\n\n\n\n\nTags\n\n\n\n\nFAQ\n\n\n\n\nemojisearch.app\n\n\n\n\n\n\n\n\n\n      Prompt Engineering\n    \nDate: March 15, 2023  |  Estimated Reading Time: 21 min  |  Author: Lilian Weng\n\n\n \n\n\nTable of Contents\n\n\n\nBasic Prompting\n\nZero-Shot\n\nFew-shot\n\nTips for Example Selection\n\nTips for Example Ordering\n\n\n\nInstruction Prompting\n\nSelf-Consistency Sampling\n\nChain-of-Thought (CoT)\n\nTypes of CoT prompts\n\nTips and Extensions\n\n\nAutomatic Prompt Design\n\nAugmented Language Models\n\nRetrieval\n\nProgramming Language\n\nExternal APIs\n\n\nCitation\n\nUseful Resources\n\nReferences")],
      'steps': ['retrieve_documents',
        'grade_document_retrieval',
        'generate_answer']}

  - Irrelevant retrieval

    .. code:: python

      example = {"input": "What is the capital of China?"}
      config = {"configurable": {"thread_id": str(uuid.uuid4())}}
      state_dict = app.invoke({"question": example["input"], "steps": []}, config)
      state_dict

    Ouput:

    .. code:: python

      ---GRADE: DOCUMENT 0 ISN'T RELEVANT---
      ---GRADE: DOCUMENT 1 ISN'T RELEVANT---
      ---GRADE: DOCUMENT 2 ISN'T RELEVANT---
      ---GRADE: DOCUMENT 3 ISN'T RELEVANT---
      {'question': 'What is the capital of China?',
      'generation': '{\n      "answer": "Beijing is the capital of China."\n    }',
      'search': 'Yes',
      'documents': [Document(metadata={'url': 'https://clintonwhitehouse3.archives.gov/WH/New/China/beijing.html'}, page_content='The modern day capital of China is Beijing (literally "Northern Capital"), which first served as China\'s capital city in 1261, when the Mongol ruler Kublai\xa0...'),
        Document(metadata={'url': 'https://en.wikipedia.org/wiki/Beijing'}, page_content="Beijing, previously romanized as Peking, is the capital city of China. With more than 22 million residents, it is the world's most populous national capital\xa0..."),
        Document(metadata={'url': 'https://pubmed.ncbi.nlm.nih.gov/38294063/'}, page_content='Supercritical and homogenous transmission of monkeypox in the capital of China. J Med Virol. 2024 Feb;96(2):e29442. doi: 10.1002/jmv.29442. Authors. Yunjun\xa0...'),
        Document(metadata={'url': 'https://www.sciencedirect.com/science/article/pii/S0304387820301358'}, page_content='This paper investigates the impacts of fires on cognitive performance. We find that a one-standard-deviation increase in the difference between upwind and\xa0...')],
      'steps': ['retrieve_documents',
        'grade_document_retrieval',
        'web_search',
        'generate_answer']}


Adaptive RAG
------------

.. _fig_A_rag:
.. figure:: images/adaptive_rag.png
    :align: center

    Adaptive-RAG langgraph diagram (source `Langgraph A-rag`_)
    
.. _Langgraph A-rag: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/



Agentic RAG
-----------

.. _fig_Agentic_rag:
.. figure:: images/agentic_rag.png
    :align: center

    Agentic-RAG langgraph diagram (source `Langgraph Agentic-rag`_)
    
.. _Langgraph Agentic-rag: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/


- Load Models 

  .. code:: python

    from langchain_ollama import OllamaEmbeddings 
    from langchain_ollama.llms import OllamaLLM

    # embedding model
    embedding = OllamaEmbeddings(model="bge-m3")

    # LLM
    llm = OllamaLLM(temperature=0.0, model='mistral', format='json')  

  .. warning:: 

    You need to specify ``format='json'`` when Initializing ``OllamaLLM``. otherwise
    you will get error:

      .. figure:: images/gemini.png
        :align: center 

-  Create Index

  .. code:: python

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaEmbeddings  # Import OllamaEmbeddings instead


    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="bge-m3"),
    )
    retriever = vectorstore.as_retriever(k=4)

- Retrieval Grader

  .. code:: python
      
    ### Retrieval Grader

    from langchain_ollama.llms import OllamaLLM
    from langchain.prompts import PromptTemplate

    # Import from pydantic directly instead of langchain_core.pydantic_v1
    from pydantic import BaseModel, Field
    from langchain.output_parsers import PydanticOutputParser

    # Data model
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        score: str = Field(  # Changed field name to 'score'
            description="Documents are relevant to the question, 'yes' or 'no'")

    parser = PydanticOutputParser(pydantic_object=GradeDocuments)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved
        document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question,
        grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out
        erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document
        is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no
        premable or explanation.""",
        input_variables=["context", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    retrieval_grader = prompt | llm | parser
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    retrieval_grader.invoke({"question": question, "context": doc_txt})


  Ouput:

  .. code:: python

    GradeDocuments(score='yes')

- Agent State

  .. code:: python  

    from typing import Annotated, Sequence
    from typing_extensions import TypedDict

    from langchain_core.messages import BaseMessage

    from langgraph.graph.message import add_messages


    class AgentState(TypedDict):
        # The add_messages function defines how an update should be processed
        # Default is to replace. add_messages says "append"
        messages: Annotated[Sequence[BaseMessage], add_messages]  

- Create the Graph

  .. code:: python    

    from typing import Annotated, Literal, Sequence
    from typing_extensions import TypedDict

    from langchain import hub
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate


    from pydantic import BaseModel, Field
    from langchain_experimental.llms.ollama_functions import OllamaFunctions



    from langgraph.prebuilt import tools_condition

    ### Edges


    def grade_documents(state) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        print("---CHECK RELEVANCE---")

        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content

        scored_result = retrieval_grader.invoke({"question": question, \
                                                "context": docs})

        score = scored_result.score

        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "generate"

        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            print(score)
            return "rewrite"


    ### Nodes


    def agent(state):
        """
        Invokes the agent model to generate a response based on the current state.
        Given the question, it will decide to retrieve using the retriever tool,
        or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        print("---CALL AGENT---")
        messages = state["messages"]

        model = OllamaFunctions(model="mistral", format='json')
        model = model.bind_tools(tools)
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}


    def rewrite(state):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """

        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n
        Look at the input and try to reason about the underlying semantic intent /
        meaning. \n
        Here is the initial question:
        \n ------- \n
        {question}
        \n ------- \n
        Formulate an improved question: """,
            )
        ]

        # Grader
        response = llm.invoke(msg)
        return {"messages": [response]}


    def generate(state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")


        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}


    # print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
    # # Show what the prompt looks like
    # prompt = hub.pull("rlm/rag-prompt").pretty_print()


    from langgraph.graph import END, StateGraph, START
    from langgraph.prebuilt import ToolNode

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the nodes we will cycle between
    workflow.add_node("agent", agent)  # agent
    retrieve = ToolNode([retriever_tool])
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node(
        "generate", generate
    )  # Generating a response after we know the documents are relevant
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    # Compile
    graph = workflow.compile()


  .. warning::

    ``OllamaLLM`` object has no attribute ``bind_tools``. You need to Install ``langchain-experimental``:
    OllamaFunctions is initialized with the desired model name: 
    
    .. code:: python

      model = OllamaFunctions(model="mistral", format='json')
      model = model.bind_tools(tools)
      response = model.invoke(messages)

    If you use OpenAI model, the code should be like:

    .. code:: python

      model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
      model = model.bind_tools(tools)
      response = model.invoke(messages)



- Graph visualization

  .. code:: python

    from IPython.display import Image, display

    try:
        display(Image(app.get_graph(xray=True).draw_mermaid_png()))
    except:
        pass

  Ouput

  .. _fig_agentic_rag_graph:
  .. figure:: images/agentic_rag_graph.png
      :align: center

      Agentic-RAG Graph 

- Test 


  .. code:: python

    import pprint

    inputs = {
        "messages": [
            ("user", "What does Lilian Weng say about the types of agent memory?"),
        ]
    }
    for output in graph.stream(inputs):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n") 

  Ouput:

  .. code:: python


    ---CALL AGENT---
    "Output from node 'agent':"
    '---'
    { 'messages': [ AIMessage(content='', additional_kwargs={}, response_metadata={}, id='run-ba7e9f54-7b32-44be-a39b-b083a6db462d-0', tool_calls=[{'name': 'retrieve_blog_posts', 'args': {'query': 'types of agent memory'}, 'id': 'call_4b1ac43a51c545cb942498b35321693a', 'type': 'tool_call'}])]}
    '\n---\n'
    ---CHECK RELEVANCE---
    ---DECISION: DOCS RELEVANT---
    "Output from node 'retrieve':"
    '---'
    { 'messages': [ ToolMessage(content='Fig. 7. Comparison of AD, ED, source policy and RL^2 on environments that require memory and exploration. Only binary reward is assigned. The source policies are trained with A3C for "dark" environments and DQN for watermaze.(Image source: Laskin et al. 2023)\nComponent Two: Memory#\n(Big thank you to ChatGPT for helping me draft this section. I’ve learned a lot about the human brain and data structure for fast MIPS in my conversations with ChatGPT.)\nTypes of Memory#\nMemory can be defined as the processes used to acquire, store, retain, and later retrieve information. There are several types of memory in human brains.\n\n\nSensory Memory: This is the earliest stage of memory, providing the ability to retain impressions of sensory information (visual, auditory, etc) after the original stimuli have ended. Sensory memory typically only lasts for up to a few seconds. Subcategories include iconic memory (visual), echoic memory (auditory), and haptic memory (touch).\n\nShort-term memory: I would consider all the in-context learning (See Prompt Engineering) as utilizing short-term memory of the model to learn.\nLong-term memory: This provides the agent with the capability to retain and recall (infinite) information over extended periods, often by leveraging an external vector store and fast retrieval.\n\n\nTool use\n\nThe agent learns to call external APIs for extra information that is missing from the model weights (often hard to change after pre-training), including current information, code execution capability, access to proprietary information sources and more.\n\nSensory memory as learning embedding representations for raw inputs, including text, image or other modalities;\nShort-term memory as in-context learning. It is short and finite, as it is restricted by the finite context window length of Transformer.\nLong-term memory as the external vector store that the agent can attend to at query time, accessible via fast retrieval.\n\nMaximum Inner Product Search (MIPS)#\nThe external memory can alleviate the restriction of finite attention span.  A standard practice is to save the embedding representation of information into a vector store database that can support fast maximum inner-product search (MIPS). To optimize the retrieval speed, the common choice is the approximate nearest neighbors (ANN)\u200b algorithm to return approximately top k nearest neighbors to trade off a little accuracy lost for a huge speedup.\nA couple common choices of ANN algorithms for fast MIPS:\n\nThey also discussed the risks, especially with illicit drugs and bioweapons. They developed a test set containing a list of known chemical weapon agents and asked the agent to synthesize them. 4 out of 11 requests (36%) were accepted to obtain a synthesis solution and the agent attempted to consult documentation to execute the procedure. 7 out of 11 were rejected and among these 7 rejected cases, 5 happened after a Web search while 2 were rejected based on prompt only.\nGenerative Agents Simulation#\nGenerative Agents (Park, et al. 2023) is super fun experiment where 25 virtual characters, each controlled by a LLM-powered agent, are living and interacting in a sandbox environment, inspired by The Sims. Generative agents create believable simulacra of human behavior for interactive applications.\nThe design of generative agents combines LLM with memory, planning and reflection mechanisms to enable agents to behave conditioned on past experience, as well as to interact with other agents.\n\nMemory stream: is a long-term memory module (external database) that records a comprehensive list of agents’ experience in natural language.', name='retrieve_blog_posts', id='2ca2d54b-214b-4463-a491-20c8d08e79cc', tool_call_id='call_4b1ac43a51c545cb942498b35321693a')]}
    '\n---\n'
    ---GENERATE---
    /usr/local/lib/python3.10/dist-packages/langsmith/client.py:261: LangSmithMissingAPIKeyWarning: API key must be provided when using hosted LangSmith API
      warnings.warn(
    "Output from node 'generate':"
    '---'
    { 'messages': [ '{\n'
                    '   "Lilian Weng describes three types of memory: Sensory '
                    'Memory, Short-Term Memory, and Long-Term Memory. Sensory '
                    'Memory is the earliest stage, lasting for up to a few '
                    'seconds, and includes iconic (visual), echoic (auditory), and '
                    'haptic memory. Short-Term Memory is used for in-context '
                    'learning and is finite due to the limited context window '
                    'length of Transformer. Long-Term Memory provides agents with '
                    'the capability to retain and recall information over extended '
                    'periods by leveraging an external vector store and fast '
                    "retrieval.}'{: .language-json }. In the given context, it "
                    'does not explicitly mention any specific agent memory types '
                    'related to reinforcement learning or simulation experiments. '
                    'For those topics, you may want to refer to the sections on "\n'
                    '\n'
                    ' \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t']}           
