
.. _prompt:

==================
Prompt Engineering 
==================




Prompt
++++++

A prompt is the input or query given to an LLM to elicit a specific response.
It acts as the user's way of "programming" the model without code, simply by 
phrasing questions or tasks appropriately.


Prompt Engineering
++++++++++++++++++

What's Prompt Engineering
-------------------------

Prompt engineering is the practice of designing and refining input prompts to 
guide LLMs to produce desired outputs effectively and consistently. 
It involves crafting queries, commands, or instructions that align with 
the model's capabilities and the task's requirements.


Key Elements of a Prompt
------------------------

- Clarity: A clear and unambiguous prompt ensures the model understands the task.
- Specificity: Including details like tone, format, length, or audience helps tailor the response.
- Context:Providing background information ensures the model generates relevant outputs.



Advanced Prompt Engineering
+++++++++++++++++++++++++++

Role Assignment
---------------

- Assign a specific role or persona to the AI to shape its style and expertise.
- **Example:**  

    *"You are a professional data scientist. Explain how to build a machine learning model to a beginner."*

.. code-block:: python 

    # You are a professional data scientist. Explain how to build a machine
    # learning model to a beginner.
    template = """Role: you are a {role}
    task: {task}
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(temperature=0.0, model=MODEL, format='json')
    output_parser = CommaSeparatedListOutputParser()

    chain = prompt | model | output_parser

    response = chain.invoke({'role': 'data scientist', \
                                "task": "Explain how to build a machine \
                                learning model to a beginner"})
    print(response)

.. code-block:: python

    {
   "role": "assistant",
   "content": "To build a machine learning model, let's follow these steps as a beginner: \n\n1. **Define the Problem**: Understand what problem you are trying to solve. This could be anything from predicting house prices, recognizing images, or even recommending products. \n\n2. **Collect and Prepare Data**: Gather relevant data for your problem. This might involve web scraping, APIs, or using existing datasets. Once you have the data, clean it by handling missing values, outliers, and errors. \n\n3. **Explore and Visualize Data**: Understand the structure of your data, its distribution, and relationships between variables. This can help in identifying patterns and making informed decisions about the next steps. \n\n4. **Feature Engineering**: Create new features that might be useful for the model to make accurate predictions. This could involve creating interactions between existing features or using techniques like one-hot encoding. \n\n5. **Split Data**: Split your data into training, validation, and testing sets. The training set is used to train the model, the validation set is used to tune hyperparameters, and the testing set is used to evaluate the final performance of the model. \n\n6. **Choose a Model**: Select a machine learning algorithm that suits your problem. Some common algorithms include linear regression for regression problems, logistic regression for binary classification problems, decision trees, random forests, support vector machines (SVM), and neural networks for more complex tasks. \n\n7. **Train the Model**: Use your training data to train the chosen model. This involves feeding the data into the model and adjusting its parameters based on the error it makes. \n\n8. **Tune Hyperparameters**: Adjust the hyperparameters of the model to improve its performance. This could involve changing learning rates, number of layers in a neural network, or the complexity of a decision tree. \n\n9. **Evaluate the Model**: Use your testing data to evaluate the performance of the model. Common metrics include accuracy for classification problems, mean squared error for regression problems, and precision, recall, and F1 score for imbalanced datasets. \n\n10. **Deploy the Model**: Once you are satisfied with the performance of your model, deploy it to a production environment where it can make predictions on new data."
    }

Contextual Setup
----------------

- Provide sufficient background or context for the AI to understand the task.
- **Example:**  

    *"I am planing to write a book about GenAI best practice, help me draft the contents for the book."*

.. code-block:: python

    # Contextual Setup

    # I am planing to write a book about GenAI best practice, help me draft the
    # contents for the book.
    template = """Role: you are a {role}
    task: {task}
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(temperature=0.0, model=MODEL, format='json')
    output_parser = CommaSeparatedListOutputParser()

    chain = prompt | model | output_parser

    response = chain.invoke({'role': 'book writer', \
                            "task": "I am planing to write a book about \
                            GenAI best practice, help me draft the \
                            contents for the book."})
    print(response)



.. code-block:: python

    {"1. Introduction": "Introduction to General Artificial Intelligence (GenAI) and its significance in today's world.",
    "2. Chapter 1 - Understanding AI": "Exploring the basics of Artificial Intelligence, its history, and evolution.",
    "3. Chapter 2 - Types of AI": "Detailed discussion on various types of AI such as Narrow AI, General AI, and Superintelligent AI.",
    "4. Chapter 3 - GenAI Architecture": "Exploring the architecture of General AI systems, including neural networks, deep learning, and reinforcement learning.",
    "5. Chapter 4 - Ethics in AI Development": "Discussing the ethical considerations involved in developing GenAI, such as privacy, bias, and accountability.",
    "6. Chapter 5 - Data Collection and Management": "Understanding the importance of data in AI development, best practices for data collection, and responsible data management.",
    "7. Chapter 6 - Model Training and Optimization": "Exploring techniques for training AI models effectively, including hyperparameter tuning, regularization, and optimization strategies.",
    "8. Chapter 7 - Testing and Validation": "Discussing the importance of testing and validation in ensuring the reliability and accuracy of GenAI systems.",
    "9. Chapter 8 - Deployment and Maintenance": "Exploring best practices for deploying AI models into production environments, as well as ongoing maintenance and updates.",
    "10. Case Studies": "Real-world examples of successful GenAI implementations across various industries, highlighting key takeaways and lessons learned.",
    "11. Future Trends in GenAI": "Exploring emerging trends in the field of General AI, such as quantum computing, explainable AI, and human-AI collaboration.",
    "12. Conclusion": "Summarizing the key points discussed in the book and looking forward to the future of General AI."}

Explicit Instructions
---------------------

- Clearly specify the format, tone, style, or structure you want in the response.
- **Example:**  

    *"Explain the concept of word embeddings in 100 words, using simple language suitable for a high school student."*

.. code-block:: python

    # Explicit Instructions
    from langchain_ollama.llms import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.output_parsers import CommaSeparatedListOutputParser


    # Explain the concept of word embeddings in 100 words, using simple 
    # language suitable for a high school student

    template = """you are a {role}
    task: {task}
    instruction: {instruction}
    Answer: Let's think step by step.
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(temperature=0.0, model=MODEL, format='json')
    output_parser = CommaSeparatedListOutputParser()

    chain = prompt | model 

    response = chain.invoke({'role': 'AI engineer', \
                            'task': "Explain the concept of word embeddings in \
                                    100 words",\
                            'instruction': "using simple \
                                    language suitable for a high school student"})
                
    print(response)


.. code-block:: python

    {
    "assistant": {
        "message": "Word Embeddings are like giving words a special address in a big library. Each word gets its own unique location, and words that are used in similar ways get placed close together. This helps the computer understand the meaning of words better when it's reading text. For example, 'king' might be near 'queen', because they are both types of royalty. And 'apple' might be near 'fruit', because they are related concepts."
    }
    }



Chain of Thought (CoT) Prompting
--------------------------------

- Encourage step-by-step reasoning for complex problems.
- **Example:**  

    *"Solve this math problem step by step: A train travels 60 miles in 1.5 hours. What is its average speed?"*

.. code-block:: python

    # CoT
    from langchain_ollama.llms import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.output_parsers import CommaSeparatedListOutputParser


    # Solve this math problem step by step: A train travels 60 miles in 1.5 hours. 
    # What is its average speed?

    template = """you are a {role}
    task: {task}
    question: {question}
    Answer: Let's think step by step.
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(temperature=0.0, model=MODEL, format='json')
    output_parser = CommaSeparatedListOutputParser()

    chain = prompt | model 

    response = chain.invoke({'role': 'math student', \
                            'task': "Solve this math problem step by step: \
                                    A train travels 60 miles in 1.5 hours.",\
                            'question': "What is its average speed per minute?"})
                
    print(response)

.. code-block:: python

    {
   "Solution": {
      "Step 1": "First, let's find the average speed of the train per hour.",
      "Step 2": "The train travels 60 miles in 1.5 hours. So, its speed per hour is 60 miles / 1.5 hours = 40 miles/hour.",
      "Step 3": "Now, let's find the average speed of the train per minute. Since there are 60 minutes in an hour, the speed per minute would be the speed per hour multiplied by the number of minutes in an hour divided by 60.",
      "Step 4": "So, the average speed of the train per minute is (40 miles/hour * (1 hour / 60)) = (40/60) miles/minute = 2/3 miles/minute."
                }
    }


Few-Shot Prompting
------------------

- Provide examples to guide the AI on how to respond.
- **Example:**  

    *"Here are examples of loan application decision:  
     'example': {'input': {'fico':800, 'income':100000,'loan_amount': 10000} 
     'decision': "accept" 
     Now Help me to make a decision to accpet or reject the loan application and 
     give the reason.
     'input': "{'fico':820, 'income':100000, 'loan_amount': 1,000}""*

.. code-block:: python

    # Few-Shot Prompting
    from langchain_ollama.llms import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.output_parsers import CommaSeparatedListOutputParser


    # Here are examples of loan application decision:  
    # 'example': {'input': {'fico':800, 'income':100000,'loan_amount': 10000} 
    # 'decision': "accept" 
    # Now Help me to make a decision to accpet or reject the loan application and 
    # give the reason.
    # 'input': "{'fico':820, 'income':100000, 'loan_amount': 1,000}"

    template = """you are a {role}
    task: {task}
    examples: {example}
    input: {input} 
    decision: 
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(temperature=0.0, model=MODEL, format='json')
    output_parser = CommaSeparatedListOutputParser()

    chain = prompt | model 

    response = chain.invoke({'role': 'banker', \
                            'task': "Help me to make a decision to accpet or \
                                    reject the loan application ",\
                            'example': {'input': {'fico':800, 'income':100000,\
                                                'loan_amount': 10000},\
                                        'decision': "accept"}, \
                            'input': {'fico':820, 'income':100000, \
                                        'loan_amount': 1000}   
                            })
                
    print(response)

.. code-block:: python  

    {"decision": "accept"}

Iterative Prompting
-------------------

- Build on the AI's response by asking follow-up questions or refining the output.
- **Example:**  

    - *Initial Prompt:* " Help me to make a decision to accpet or reject the loan application."  
    - *Follow-Up:* "give me the reason"  

.. code-block:: python

    # Few-Shot Prompting
    from langchain_ollama.llms import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.output_parsers import CommaSeparatedListOutputParser


    # Here are examples of loan application decision:  
    # 'example': {'input': {'fico':800, 'income':100000,'loan_amount': 10000} 
    # 'decision': "accept" 
    # Now Help me to make a decision to accpet or reject the loan application and 
    # give the reason.
    # 'input': "{'fico':820, 'income':100000, 'loan_amount': 1,000}"

    template = """you are a {role}
    task: {task}
    examples: {example}
    input: {input} 
    decision: 
    reason:
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(temperature=0.0, model=MODEL, format='json')
    output_parser = CommaSeparatedListOutputParser()

    chain = prompt | model 

    response = chain.invoke({'role': 'banker', \
                            'task': "Help me to make a decision to accpet or \
                                    reject the loan application and \
                                    give the reason.",\
                            'example': {'input': {'fico':800, 'income':100000,\
                                                'loan_amount': 10000},\
                                        'decision': "accept"}, \
                            'input': {'fico':820, 'income':100000, \
                                        'loan_amount': 1000}   
                            })
                
    print(response)

.. code-block:: python  

    {"decision": "accept", "reason": "The applicant has a high credit score (FICO 820), a stable income of $100,000, and is requesting a relatively small loan amount ($1000). These factors indicate a low risk for the bank."}  


Instructional Chaining
----------------------

   - Break down a task into a sequence of smaller prompts.
   - **Example:**  

     - step 1: check the fico score
     - step 2: check the income,
     - step 3: check the loan amount,
     - step 4: make a decision,
     - step 5: give the reason.

.. code-block:: python  

    # Instructional Chaining
    from langchain_ollama.llms import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.output_parsers import CommaSeparatedListOutputParser


    # Now Help me to make a decision to accpet or reject the loan application and
    # give the reason.
    # ''input': {'fico':320, 'income':10000, 'loan_amount': 100000}

    template = """you are a {role}
    task: {task}
    instruction: {instruction}
    input: {input}
    decision:
    reason:
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(temperature=0.0, model=MODEL, format='json')
    output_parser = CommaSeparatedListOutputParser()

    chain = prompt | model

    response = chain.invoke({'role': 'banker', \
                            'task': "Help me to make a decision to accpet or \
                                    reject the loan application and \
                                    give the reason.",\
                            'instruction': {'step 1': "check the fico score",\
                                            'step 2': "check the income",\
                                            'step 3': "check the loan amount",\
                                            'step 4': "make a decision",\
                                            'step 5': "give the reason"
                                            },
                            'input': {'fico':320, 'income':10000, \
                                        'loan_amount': 100000}
                            })

    print(response)

.. code-block:: python      

   {
      "decision": "reject",
      "reason": "Based on the provided information, the applicant's FICO score is 320 which falls below our minimum acceptable credit score. Additionally, the proposed loan amount of $100,000 exceeds the income level of $10,000 per year, making it difficult for the borrower to repay the loan."
   }

Use Constraints
---------------

   - Impose constraints to keep responses concise and on-topic.
   - **Example:**  

     *"List 5 key trends in AI in bullet points, each under 15 words."*

Creative Prompting
------------------

   - Encourage unique or unconventional ideas by framing the task creatively.
   - **Example:**  

     *"Pretend you are a time traveler from the year 2124. How would you describe AI advancements to someone today?"*

Feedback Incorporation
----------------------

    - If the response isn’t perfect, guide the AI to refine or retry.
    - **Example:**  

      *"This is too general. Could you provide more specific examples for the education industry?"*

Scenario-Based Prompts
----------------------

    - Frame the query within a scenario for a contextual response.
    - **Example:**  

      *"Imagine you're a teacher explaining ChatGPT to students. How would you introduce its uses and limitations?"*

Multimodal Prompting
--------------------

    - Use prompts designed for mixed text/image inputs (or outputs if using models like DALL·E).
    - **Example:**  

      *"Generate an image prompt for a futuristic cityscape, vibrant, with flying cars and greenery."*