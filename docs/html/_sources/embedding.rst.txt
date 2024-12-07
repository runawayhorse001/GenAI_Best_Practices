
.. _embedding:

===========================
Word and Sentence Embedding
===========================

Bag-of-Word
+++++++++++

**Bag of Words (BoW)** is a simple and widely used text representation technique in natural language processing (NLP). It represents a text (e.g., a document or a sentence) as a collection of words, ignoring grammar, order, and context but keeping their frequency.

Key Features of Bag of Words:

1. **Vocabulary Creation**:
   - A list of all unique words in the dataset (the "vocabulary") is created.
   - Each word becomes a feature.

2. **Representation**:
   - Each document is represented as a vector or a frequency count of words from the vocabulary.
   - If a word from the vocabulary is present in the document, its count is included in the vector.
   - Words not present in the document are assigned a count of zero.

3. **Simplicity**:
   - The method is computationally efficient and straightforward.
   - However, it ignores the sequence and semantic meaning of the words.

Applications:

- Text Classification
- Sentiment Analysis
- Document Similarity

Limitations:

1. **Context Ignorance**:
   - BoW does not capture word order or semantics.
   - For example, "not good" and "good" might appear similar in BoW.

2. **Dimensionality**:
   - As the vocabulary size increases, the vector representation grows, leading to high-dimensional data.

3. **Sparse Representations**:
   - Many entries in the vectors might be zeros, leading to sparsity.


 One Hot Encoder
----------------

CountVectorizer
---------------



To overcome these limitations, advanced techniques like **TF-IDF**, **word embeddings** (e.g., Word2Vec, GloVe), and contextual embeddings (e.g., BERT) are often used.

TF-IDF
++++++


Word2Vec
++++++++

GloVE
+++++

Fast Text 
+++++++++

