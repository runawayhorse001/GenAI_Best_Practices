
.. _rag:

==============================
Retrieval-Augmented Generation
==============================


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

    Naive chunking v.s. late chunking (Souce `Jina AI`_)

.. _Jina AI: https://www.marktechpost.com/2024/08/27/jina-ai-introduced-late-chunking-a-simple-ai-approach-to-embed-short-chunks-by-leveraging-the-power-of-long-context-embedding-models/

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

Common retrieval methods
------------------------

- Sparse Vector Search: Traditional keyword-based retrieval (e.g., TF-IDF, BM25).
- Dense Vector Search: Vector-based search using embeddings e.g.

  - Approximate Nearest Neighbor (ANN) Search: 
     - HNSW (Hierarchical Navigable Small World): Graph-based approach
     - IVF (Inverted File Index): Clusters embeddings into groups and searches within relevant clusters.
  - Exact Nearest Neighbor Search: Computes similarities exhaustively for all vectors in the corpus    

- Hybrid Search: the combination of Sparse and Dense vector search. 

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


