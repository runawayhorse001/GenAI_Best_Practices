
.. _finetuning:

===========
Fine Tuning
===========

.. admonition:: Chinese proverb

	Good tools are prerequisite to the successful execution of a job. – old Chinese proverb

.. admonition:: Colab Notebook for This Chapter

    - Embedding Model Fine-tuning: |e_tune|
    - LLM (Llama 2 7B) Model Fine-tuning: |l_tune|
    
    .. |e_tune| image:: images/colab-badge.png 
        :target: https://colab.research.google.com/drive/14aYT8Ydm_e-z47yGpctAfk246K_PK1LC?usp=drive_link  

    .. |l_tune| image:: images/colab-badge.png 
        :target: https://colab.research.google.com/drive/1GPu2vNRdcObf0dmP7r_M42NYx0OXVD_F?usp=drive_link  


Fine-tuning is a machine learning technique where a pre-trained model (like a large 
language model or neural network) is further trained on a smaller, specific dataset
to adapt it to a particular task or domain. Instead of training a model from scratch, 
fine-tuning leverages the knowledge already embedded in the pre-trained model, 
saving time, computational resources, and data requirements.


.. _fig_fine_tuning:
.. figure:: images/fine_tuning.png
    :align: center

    The three conventional feature-based and finetuning approaches (Souce `Finetuning Sebastian`_).

.. _Finetuning Sebastian: https://magazine.sebastianraschka.com/p/finetuning-large-language-models

Cutting-Edge Strategies for LLM Fine-Tuning
+++++++++++++++++++++++++++++++++++++++++++

Over the past year, fine-tuning methods have made remarkable strides. Modern methods 
for fine-tuning LLMs focus on efficiency, scalability, and resource optimization. 
The following strategies are at the forefront:

LoRA (Low-Rank Adaptation)
--------------------------

**LoRA** reduces the number of trainable parameters by introducing **low-rank decomposition** into the fine-tuning process.

.. _fig_lora:
.. figure:: images/lora.png
    :align: center

    Weight update matrix (Souce `LORA Sebastian`_).

.. _LORA Sebastian: https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms


**How It Works**:  

- Instead of updating all model weights, LoRA injects **low-rank adapters** into the model’s layers.  
- The original pre-trained weights remain frozen; only the low-rank parameters are optimized.

**Benefits**:  

- Reduces memory and computational requirements.  
- Enables fine-tuning on resource-constrained hardware.

QLoRA (Quantized Low-Rank Adaptation)
-------------------------------------

**QLoRA** combines **low-rank adaptation** with **4-bit quantization** of the pre-trained model.

**How It Works**:  

- The LLM is quantized to **4-bit precision** to reduce memory usage.  
- LoRA adapters are applied to the quantized model for fine-tuning.  
- Precision is maintained using methods like **NF4 (Normalized Float 4)** and double backpropagation.

**Benefits**:  

- Further reduces memory usage compared to LoRA.  
- Enables fine-tuning of massive models on consumer-grade GPUs.

PEFT (Parameter-Efficient Fine-Tuning)
--------------------------------------

**PEFT** is a general framework for fine-tuning LLMs with minimal trainable parameters.

.. list-table::
   :width: 100%
   :class: borderless

   * - .. image:: images/peft_1.png
        :width: 100%
            
     - .. image:: images/peft_2.png
        :width: 100%

Source: [PEFT]_ 

**Techniques Under PEFT**:

- **LoRA**: Low-rank adaptation of weights.  
- **Adapters**: Small trainable layers inserted into the model.  
- **Prefix Tuning**: Fine-tuning input prefixes instead of weights.  
- **Prompt Tuning**: Optimizing soft prompts in the input space.

**Benefits**:  

- Reduces the number of trainable parameters.  
- Faster training and lower hardware requirements.

SFT (Supervised Fine-Tuning)
----------------------------

**SFT** adapts an LLM using a labeled dataset in a fully supervised manner.

**How It Works**:  

- The model is initialized with pre-trained weights.  
- It is fine-tuned on a task-specific dataset with a supervised loss function (e.g., cross-entropy).

**Benefits**:  

- Achieves high performance on specific tasks.  
- Essential for aligning models with labeled datasets.

RLHF (Reinforcement Learning from Human Feedback)
-------------------------------------------------

**RLHF** is a technique used to fine-tune language models, aligning their 
behavior with human preferences or specific tasks. RLHF incorporates feedback 
from humans to guide the model's learning process, ensuring that its outputs 
are not only coherent but also align with desired ethical, practical, or 
stylistic goals.

**How It Works**:  

- The model is initialized with pre-trained weights.  
- The pretrained model is fine-tuned further using reinforcement learning, 
  guided by the reward model.
- A reinforcement learning algorithm, such as Proximal Policy Optimization 
  (PPO), optimizes the model to maximize the reward assigned by the reward model.

.. note::

    - **Direct Preference Optimization** 
        DPO is a technique for aligning large 
        language models (LLMs) with human preferences, offering an alternative 
        to the traditional Reinforcement Learning from Human Feedback (RLHF) 
        approach that uses Proximal Policy Optimization (PPO). Instead of 
        training a separate reward model and using reinforcement learning, 
        DPO simplifies the process by directly leveraging human preference 
        data to fine-tune the model through supervised learning.

    - **Proximal Policy Optimization** 
        PPO is a reinforcement learning algorithm 
        commonly used in RLHF to fine-tune LLMs. PPO optimizes the model's policy 
        by maximizing the reward signal provided by a reward model, which 
        represents human preferences.

    - **Comparison: DPO vs PPO**


        +---------------------------+-----------------------------------------+----------------------------------------+
        | **Feature**               | **DPO**                                 | **PPO**                                |
        +===========================+=========================================+========================================+
        | **Training Paradigm**     | Supervised fine-tuning with preferences | Reinforcement learning with a reward   |
        |                           |                                         | model                                  |
        +---------------------------+-----------------------------------------+----------------------------------------+
        | **Workflow Complexity**   | Simpler                                 | More complex (requires reward model    |
        |                           |                                         | and iterative RL)                      |
        +---------------------------+-----------------------------------------+----------------------------------------+
        | **Stability**             | More stable (uses supervised learning)  | Less stable (inherent to RL methods)   |
        +---------------------------+-----------------------------------------+----------------------------------------+
        | **Efficiency**            | Computationally efficient               | Computationally intensive              |
        +---------------------------+-----------------------------------------+----------------------------------------+
        | **Scalability**           | Scales well with large preference       | Requires significant compute for RL    |
        |                           | datasets                                | steps                                  |
        +---------------------------+-----------------------------------------+----------------------------------------+
        | **Use Case**              | Directly aligns LLM with preferences    | Optimizes policy for long-term reward  |
        |                           |                                         | maximization                           |
        +---------------------------+-----------------------------------------+----------------------------------------+
        | **Human Preference        | Directly encoded in loss function       | Encoded via a reward model             |
        | Modeling**                |                                         |                                        |
        +---------------------------+-----------------------------------------+----------------------------------------+        



**Benefits**:  

- RLHF ensures the model's outputs are ethical, safe, and aligned with human 
  expectations, reducing harmful or biased content. 
- Responses become more relevant, helpful, and contextually appropriate, 
  enhancing user experience.
- Fine-tuning with RLHF allows models to be customized for specific use cases, 
  such as customer service, creative writing, or technical support.


Summary Table
-------------

+-------------------+---------------------------------------------+--------------------------------------------+
| **Method**        | **Description**                             | **Key Benefit**                            |
+-------------------+---------------------------------------------+--------------------------------------------+
| **LoRA**          | Low-rank adapters for parameter-efficient   | Reduces trainable parameters significantly.|
|                   | tuning.                                     |                                            |
+-------------------+---------------------------------------------+--------------------------------------------+
| **QLoRA**         | LoRA with 4-bit quantization of the model.  | Fine-tunes massive models on smaller       |
|                   |                                             | hardware.                                  |
+-------------------+---------------------------------------------+--------------------------------------------+
| **PEFT**          | General framework for efficient fine-tuning.| Includes LoRA, Adapters, Prefix Tuning,    |
|                   |                                             | etc.                                       |
+-------------------+---------------------------------------------+--------------------------------------------+
| **SFT**           | Supervised fine-tuning with labeled data.   | High performance on task-specific datasets |
+-------------------+---------------------------------------------+--------------------------------------------+


These strategies represent the forefront of **LLM fine-tuning**, offering efficient and scalable solutions for 
real-world applications. To choose the most suitable strategy, consider the following factors:

- **Resource-Constrained Environments**: Use **LoRA** or **QLoRA**.  
- **Large-Scale Models**: **QLoRA** for low-memory fine-tuning.  
- **High Performance with Labeled Data**: **SFT**.  
- **Minimal Setup**: **Zero-shot** or **Few-shot** learning.  
- **General Efficiency**: Use **PEFT** frameworks.


Key Early Fine-Tuning Methods
+++++++++++++++++++++++++++++

Early fine-tuning methods laid the foundation for current approaches. These methods 
primarily focused on updating the entire model or selected components.

Full Fine-Tuning
----------------

All the parameters of a pre-trained model are updated using task-specific data :ref:`fig_fine_tuning` (right).

**How It Works**:  

- The pre-trained model serves as the starting point.  
- Fine-tuning is conducted on a smaller, labeled dataset using a supervised loss function.  
- A low learning rate is used to prevent **catastrophic forgetting**.

**Benefits**:  

- Effective at adapting models to specific tasks.  

**Challenges**:  

- Computationally expensive.  
- Risk of overfitting on small datasets.

Feature-Based Approach
----------------------

The pre-trained model is used as a **feature extractor**, while only a task-specific head is trained :ref:`fig_fine_tuning` (left).

**How It Works**:  

- The model processes inputs and extracts features (embeddings).  
- A separate classifier (e.g., linear or MLP) is trained on top of these features.  
- The pre-trained model weights remain **frozen**.

**Benefits**:  

- Computationally efficient since only the task-specific head is trained.  


Layer-Specific Fine-Tuning
--------------------------

Only certain layers of the pre-trained model are fine-tuned while the rest remain frozen :ref:`fig_fine_tuning` (middle).

**How It Works**:  

- Earlier layers (which capture general features) are frozen.  
- Later layers (closer to the output) are fine-tuned on task-specific data.  

**Benefits**:  

- Balances computational efficiency and task adaptation.  


Task-Adaptive Pre-training
--------------------------

Before fine-tuning on a specific task, the model undergoes additional **pre-training** on a domain-specific corpus.

**How It Works**: 

- A general pre-trained model is further pre-trained (unsupervised) on domain-specific data.  
- Fine-tuning is then performed on the downstream task.

**Benefits**:  

- Provides a better starting point for domain-specific tasks.  



Embedding Model Fine-Tuning
+++++++++++++++++++++++++++

In the chapter :ref:`rag`, we discussed how embedding models are crucial for the success of RAG applications. 
However, their general-purpose training often limits their effectiveness for company- or domain-specific 
use cases. Customizing embeddings with domain-specific data can significantly improve the retrieval 
performance of your RAG application.

In this chapter, we will demonstrate how to fine-tune embedding models using the 
``SentenceTransformersTrainer``, building on insights shared in the blog [fineTuneEmbedding]_ and 
Sentence Transformer `Training Overview`_. Our main contribution was introducing LoRA to enable functionality on 
NVIDIA T4 GPUs, while the rest of the pipeline and code remained almost unchanged.

.. _`Training Overview`: https://sbert.net/docs/sentence_transformer/training_overview.html#dataset-format

.. note::

    Please ensure that the package versions are set as follows:
    
    .. code-block:: python 

        pip install  "torch==2.1.2" tensorboard
        
        pip install --upgrade \
            sentence-transformers>=3 \
            datasets==2.19.1  \
            transformers==4.41.2 \
            peft==0.10.0

    Otherwise, you may encounter the error.  


Prepare Dataset
---------------

We are going to directly use the synthetic dataset ``philschmid/finanical-rag-embedding-dataset``, which includes 7,000 
positive text pairs of questions and corresponding context from the `2023_10 NVIDIA SEC Filing`_.

.. _2023_10 NVIDIA SEC Filing: https://stocklight.com/stocks/us/nasdaq-nvda/nvidia/annual-reports/nasdaq-nvda-2023-10K-23668751.pdf

.. code-block:: python 

    from datasets import load_dataset

    # Load dataset from the hub
    dataset = load_dataset("philschmid/finanical-rag-embedding-dataset", split="train")

    # rename columns
    dataset = dataset.rename_column("question", "anchor")
    dataset = dataset.rename_column("context", "positive")

    # Add an id column to the dataset
    dataset = dataset.add_column("id", range(len(dataset)))

    # split dataset into a 10% test set
    dataset = dataset.train_test_split(test_size=0.1)

    # save datasets to disk
    dataset["train"].to_json("train_dataset.json", orient="records")
    dataset["test"].to_json("test_dataset.json", orient="records")    


.. note::

    In practice, most dataset configurations will take one of four forms:

    - **Positive Pair**: A pair of related sentences. This can be used both for symmetric tasks
      (semantic textual similarity) or asymmetric tasks (semantic search), with examples 
      including pairs of paraphrases, pairs of full texts and their summaries, pairs of 
      duplicate questions, pairs of ``(query, response)``, or pairs of 
      ``(source_language, target_language)``. 
      Natural Language Inference datasets can also be formatted this way by pairing entailing sentences.
    - **Triplets**: ``(anchor, positive, negative)`` text triplets. These datasets don't need labels.
    - **Pair with Similarity Score**: A pair of sentences with a score indicating their similarity. 
      Common examples are "Semantic Textual Similarity" datasets.
    - **Texts with Classes**: A text with its corresponding class. This data format is easily 
      converted by loss functions into three sentences (triplets) where the first is an "anchor", 
      the second a "positive" of the same class as the anchor, and the third a "negative" of a different class.

    Note that it is often simple to transform a dataset from one format to another, such that it works with 
    your loss function of choice.

Import and Evaluate Pretrained Baseline Model
---------------------------------------------

.. code-block:: python 

    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import (
        InformationRetrievalEvaluator,
        SequentialEvaluator,
    )
    from sentence_transformers.util import cos_sim
    from datasets import load_dataset, concatenate_datasets
    from peft import LoraConfig, TaskType

    model_id =  "BAAI/bge-base-en-v1.5"
    matryoshka_dimensions = [768, 512, 256, 128, 64] # Important: large to small

    # Load a model
    model = SentenceTransformer(
        model_id,
        trust_remote_code=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # load test dataset
    test_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
    train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")
    corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

    # Convert the datasets to dictionaries
    corpus = dict(
        zip(corpus_dataset["id"], corpus_dataset["positive"])
    )  # Our corpus (cid => document)
    queries = dict(
        zip(test_dataset["id"], test_dataset["anchor"])
    )  # Our queries (qid => question)

    # Create a mapping of relevant document (1 in our case) for each query
    relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
    for q_id in queries:
        relevant_docs[q_id] = [q_id]


    matryoshka_evaluators = []
    # Iterate over the different dimensions
    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # Truncate the embeddings to a certain dimension
            score_functions={"cosine": cos_sim},
        )
        matryoshka_evaluators.append(ir_evaluator)

    # Create a sequential evaluator
    evaluator = SequentialEvaluator(matryoshka_evaluators)

.. note::

   If you encounter the error ``Cannot import name 'EncoderDecoderCache' from 'transformers'``, 
   ensure that the package versions are set to ``peft==0.10.0`` and ``transformers==4.37.2``.

.. code-block:: python 

    # Evaluate the model
    results = evaluator(model)

    # Print the main score
    for dim in matryoshka_dimensions:
        key = f"dim_{dim}_cosine_ndcg@10"
        print
        print(f"{key}: {results[key]}")

.. code-block:: python        

    dim_768_cosine_ndcg@10: 0.754897248109794
    dim_512_cosine_ndcg@10: 0.7549275773474213
    dim_256_cosine_ndcg@10: 0.7454714780163237
    dim_128_cosine_ndcg@10: 0.7116728650043451
    dim_64_cosine_ndcg@10: 0.6477174937632066


Loss Function with Matryoshka Representation
--------------------------------------------

.. code-block:: python   

    from sentence_transformers import SentenceTransformerModelCardData, SentenceTransformer

    # Hugging Face model ID: https://huggingface.co/BAAI/bge-base-en-v1.5
    model_id = "BAAI/bge-base-en-v1.5"

    # load model with SDPA for using Flash Attention 2
    model = SentenceTransformer(
        model_id,
        model_kwargs={"attn_implementation": "sdpa"},
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="BGE base Financial Matryoshka",
        ),
    )

    # Apply PEFT with PromptTuningConfig
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model.add_adapter(peft_config, "dense")

    # train loss
    from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

    matryoshka_dimensions = [768, 512, 256, 128, 64]  # Important: large to small
    inner_train_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(model,
                                inner_train_loss,
                                matryoshka_dims=matryoshka_dimensions)

.. note::

    Loss functions play a critical role in the performance of your fine-tuned model. 
    Sadly, there is no "one size fits all" loss function. Ideally, 
    this table should help narrow down your choice of loss function(s) by matching 
    them to your data formats.

    You can often convert one training data format into another, allowing more loss 
    functions to be viable for your scenario. For example, 

    
    +---------------------------------------------------+--------------------------------+---------------------------------------------------------------------------------------------------------------------+
    | Inputs                                            | Labels                         | Appropriate Loss Functions                                                                                          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    +===================================================+================================+=====================================================================================================================+
    |``single sentences``                               | `class`                        | ``BatchAllTripletLoss``, ``BatchHardSoftMarginTripletLoss``, ``BatchHardTripletLoss``, ``BatchSemiHardTripletLoss`` |
    +---------------------------------------------------+--------------------------------+---------------------------------------------------------------------------------------------------------------------+
    |``single sentences``                               | `none`                         | ``ContrastiveTensionLoss``, ``DenoisingAutoEncoderLoss``                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    +---------------------------------------------------+--------------------------------+---------------------------------------------------------------------------------------------------------------------+
    |``(anchor, anchor)`` pairs                         | `none`                         | ``ContrastiveTensionLossInBatchNegatives``                                                                          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    +---------------------------------------------------+--------------------------------+---------------------------------------------------------------------------------------------------------------------+
    |``(damaged_sentence, original_sentence)`` pairs    | `none`                         | ``DenoisingAutoEncoderLoss``                                                                                        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    +---------------------------------------------------+--------------------------------+---------------------------------------------------------------------------------------------------------------------+
    |``(sentence_A, sentence_B)`` pairs                 | `class`                        | ``SoftmaxLoss``                                                                                                     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    +---------------------------------------------------+--------------------------------+---------------------------------------------------------------------------------------------------------------------+
    |``(anchor, positive)`` pairs                       | `none`                         | ``MultipleNegativesRankingLoss``, ``CachedMultipleNegativesRankingLoss``, ``MultipleNegativesSymmetricRankingLoss``,| 
    |                                                   |                                | ``CachedMultipleNegativesSymmetricRankingLoss``, ``MegaBatchMarginLoss``, ``GISTEmbedLoss``, ``CachedGISTEmbedLoss``|
    +---------------------------------------------------+--------------------------------+---------------------------------------------------------------------------------------------------------------------+
    |``(anchor, positive/negative)`` pairs              | `1 if positive, 0 if negative` | ``ContrastiveLoss``, ``OnlineContrastiveLoss``                                                                      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    +---------------------------------------------------+--------------------------------+---------------------------------------------------------------------------------------------------------------------+
    |``(sentence_A, sentence_B)`` pairs                 | `float similarity score`       | ``CoSENTLoss``, ``AnglELoss``, ``CosineSimilarityLoss``                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    +---------------------------------------------------+--------------------------------+---------------------------------------------------------------------------------------------------------------------+
    |``(anchor, positive, negative)`` triplets          | `none`                         | ``MultipleNegativesRankingLoss``, ``CachedMultipleNegativesRankingLoss``, ``TripletLoss``,                          |
    |                                                   |                                | ``CachedGISTEmbedLoss``, ``GISTEmbedLoss``                                                                          |                                                                                                                                                                                                                                                           
    +---------------------------------------------------+--------------------------------+---------------------------------------------------------------------------------------------------------------------+
    |`(anchor, positive, negative_1, ..., negative_n)`` | `none`                         | ``MultipleNegativesRankingLoss``, ``CachedMultipleNegativesRankingLoss``, ``CachedGISTEmbedLoss``                   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    +---------------------------------------------------+--------------------------------+---------------------------------------------------------------------------------------------------------------------+

Fine-tune Embedding Model 
-------------------------

.. code-block:: python  

    from sentence_transformers import SentenceTransformerTrainingArguments
    from sentence_transformers.training_args import BatchSamplers

    # load train dataset again
    train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")

    # define training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir, # output directory and hugging face model ID
        num_train_epochs=4,                         # number of epochs
        per_device_train_batch_size=32,             # train batch size
        gradient_accumulation_steps=16,             # for a global batch size of 512
        per_device_eval_batch_size=16,              # evaluation batch size
        warmup_ratio=0.1,                           # warmup ratio
        learning_rate=2e-5,                         # learning rate, 2e-5 is a good value
        lr_scheduler_type="cosine",                 # use constant learning rate scheduler
        optim="adamw_torch_fused",                  # use fused adamw optimizer
        tf32=False,                                  # use tf32 precision
        bf16=False,                                  # use bf16 precision
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        eval_strategy="epoch",                      # evaluate after each epoch
        save_strategy="epoch",                      # save after each epoch
        logging_steps=10,                           # log every 10 steps
        save_total_limit=3,                         # save only the last 3 models
        load_best_model_at_end=True,                # load the best model when training ends
        metric_for_best_model="eval_dim_128_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score for the 128 dimension
        greater_is_better=True,                     # maximize the ndcg@10 score
    )

    from sentence_transformers import SentenceTransformerTrainer

    trainer = SentenceTransformerTrainer(
        model=model, # bg-base-en-v1
        args=args,  # training arguments
        train_dataset=train_dataset.select_columns(
            ["anchor", "positive"]
        ),  # training dataset
        loss=train_loss,
        evaluator=evaluator,
    )

    # start training
    trainer.train()

    # save the best model
    #trainer.save_model()
    trainer.model.save_pretrained("bge-base-finetuning")


Evaluate Fine-tuned Model
-------------------------

.. code-block:: python  

    from sentence_transformers import SentenceTransformer

    fine_tuned_model = SentenceTransformer(
        'bge-base-finetuning', device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # Evaluate the model
    results = evaluator(fine_tuned_model)

    # # COMMENT IN for full results
    # print(results)

    # Print the main score
    for dim in matryoshka_dimensions:
        key = f"dim_{dim}_cosine_ndcg@10"
        print(f"{key}: {results[key]}")


.. code-block:: python  

    dim_768_cosine_ndcg@10: 0.7650276801072632
    dim_512_cosine_ndcg@10: 0.7603951540556889
    dim_256_cosine_ndcg@10: 0.754743133407988
    dim_128_cosine_ndcg@10: 0.7205317098443929
    dim_64_cosine_ndcg@10: 0.6609117856061502
        
Results Comparison
------------------

Although we did not observe the significant performance boost reported in the original 
blog, the fine-tuned model outperformed the baseline model across all dimensions using 
only 6.3k samples and partial parameter fine-tuning. MOre details can be found as follows:

+-----------+----------+------------+-------------+
| Dimension | Baseline | Fine-tuned | Improvement |
+===========+==========+============+=============+
| 768       | 0.75490  | 0.76503    | 1.34%       |
+-----------+----------+------------+-------------+
| 512       | 0.75492  | 0.76040    | 0.73%       |
+-----------+----------+------------+-------------+
| 256       | 0.74547  | 0.75474    | 1.24%       |
+-----------+----------+------------+-------------+
| 128       | 0.71167  | 0.72053    | 1.24%       |
+-----------+----------+------------+-------------+
| 64        | 0.64772  | 0.66091    | 2.04%       |
+-----------+----------+------------+-------------+

.. _fig_wandb:
.. figure:: images/fine_tuning_wandb.png
    :align: center

    Epoch, Training Loss/steps in Wandb


LLM Fine-Tuning
+++++++++++++++


In this chapter, we will demonstrate how to fine-tune a Llama 2 model with 7 billion parameters using 
a T4 GPU with 16 GB of VRAM. Due to VRAM limitations, traditional fine-tuning is not feasible, 
making parameter-efficient fine-tuning (PEFT) techniques like LoRA or QLoRA essential. For this 
demonstration, we use QLoRA, which leverages 4-bit precision to significantly reduce VRAM consumption.

The folloing code is from notebook [fineTuneLLM]_, and the copyright belongs to the original author.


Load Dataset and Pretrained Model 
---------------------------------

.. code-block:: python  

    # Step 1 : Load dataset (you can process it here)
    dataset = load_dataset(dataset_name, split="train")

    # Step 2 :Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Step 3 :Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Step 4 :Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Step 5 :Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"    

Fine-tuning Configuration
-------------------------

.. code-block:: python  
        
    # Step 6 :Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Step 7 :Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
    )

Fine-tune model
---------------

.. code-block:: python  

    # Step 8 :Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    # Step 9 :Train model
    trainer.train()

    # Step 10 :Save trained model
    trainer.model.save_pretrained(new_model)


.. _fig_fine_tuning_llm:
.. figure:: images/fine_tuning_llm.png
    :align: center

    Llama 2 Model Fine-Tuning TensorBoard



