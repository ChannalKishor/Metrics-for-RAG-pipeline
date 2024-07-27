
# Calorie Master: Food Nutrition Recommendation System

## Table of Contents
- Introduction
- Features
- Requirements
- Setup
- Usage
- Evaluation
- Improvement Methods
- Results
- Challenges and Solutions
- Contributing
- License
- Acknowledgements

## Introduction

### Overview
Calorie Master is a Retrieval-Augmented Generation (RAG) application designed to provide users with detailed nutrition information based on food input. The system leverages OpenAI embeddings to convert food names into vectors, and Pinecone for efficient vector storage and similarity search. The interactive user interface is built using Streamlit, offering a user-friendly way for users to get precise nutritional information.

### Purpose
The purpose of this project is to demonstrate the use of RAG techniques in creating a system that can effectively retrieve and generate relevant nutrition information for various foods. By integrating advanced AI models and robust storage solutions, Calorie Master aims to provide accurate, relevant, and helpful recommendations to users concerned about their dietary intake. The project also includes a comprehensive evaluation framework to assess and improve the performance of the system, ensuring high-quality and reliable outputs.

## Features

### Retrieval and Display of Nutritional Information
Calorie Master retrieves nutritional information for a given food item and displays it in an easy-to-read format. This allows users to quickly get key nutritional details, including calories, macronutrients, and micronutrients.

### Vector Storage and Similarity Search
The application utilizes Pinecone for vector storage and similarity search. By converting food names into vectors using OpenAI embeddings, the system can efficiently search for and retrieve relevant information based on user queries.

### Interactive User Interface
Built with Streamlit, the user interface is interactive and user-friendly. Users can simply enter a food name into the input field, and the system will provide the corresponding nutritional information. The interface is designed to be intuitive, making it easy for users to get the information they need quickly and efficiently.

## Requirements

### Software and Libraries
To run the Calorie Master, you need the following software and libraries installed:

- Python 3.8+: The programming language used to develop and run the application.
- Streamlit: An open-source app framework used to create the interactive user interface.
- Pinecone: A vector database used for efficient storage and similarity search of embeddings.
- LangChain: A library to facilitate the integration of language models into the application.
- OpenAI: The API service providing the language model used for generating embeddings and responses.
- dotenv: A module to load environment variables from a `.env` file.

You can install these dependencies using the pip package manager. The specific versions of these libraries should be listed in the `requirements.txt` file in the repository.

### API Keys
You will need API keys for the following services:

- Pinecone API Key: Required to access and use the Pinecone vector database.
- OpenAI API Key: Required to access the OpenAI API for generating embeddings and responses.

## Setup

### Cloning the Repository
To get started, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/food-nutrition-rag.git
cd food-nutrition-rag
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Setting Up Environment Variables
Create a `.env` file in the root directory and add your Pinecone and OpenAI API keys:

```bash
PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key
```

### Preparing Data
Ensure you have a JSON file named `foods.json` in the root directory containing the nutritional information data. The file should follow this structure:

```json
[
  {
    "food_name": "Apple",
    "nutrition": "Calories: 95, Carbs: 25g, Fiber: 4g, Vitamin C: 14%"
  },
  {
    "food_name": "Banana",
    "nutrition": "Calories: 105, Carbs: 27g, Potassium: 422mg, Vitamin C: 17%"
  }
]
```

## Usage

### Running the Streamlit App
To run the Streamlit app, execute the following command:

```bash
streamlit run app.py
```

### User Interaction
Open your web browser and navigate to `http://localhost:8501`. Enter a food name in the input field to get the nutritional information. The system will display the nutrition details for the specified food, providing you with key information.

For example, if you enter "Apple", the system will retrieve and display the nutritional information such as "Calories: 95, Carbs: 25g, Fiber: 4g, Vitamin C: 14%". This allows users to quickly access useful dietary information about their food choices.

### Example
Start the Streamlit app:

```bash
streamlit run main.py
```

Open your web browser and go to `http://localhost:8501`. In the input field, type a food name (e.g., "Apple"). The system will display the nutritional information for the entered food.

## Evaluation

### Performance Metrics Calculation
To evaluate the performance of Calorie Master, several metrics are calculated for both the retrieval and generation components of the system.

### Retrieval Metrics
- **Context Precision:** Measures how accurately the retrieved nutritional information matches the user's query.
- **Context Recall:** Evaluates the ability to retrieve all relevant nutritional details for the user's query.
- **Context Relevance:** Assesses the relevance of the retrieved nutrition information to the user's query.
- **Context Entity Recall:** Determines the ability to recall relevant entities within the context.
- **Noise Robustness:** Tests the system's ability to handle noisy or irrelevant inputs.

### Generation Metrics
- **Faithfulness:** Measures the accuracy and reliability of the generated answers.
- **Answer Relevance:** Evaluates the relevance of the generated answers to the user's query.
- **Information Integration:** Assesses the ability to integrate and present information cohesively.
- **Counterfactual Robustness:** Tests the robustness of the system against counterfactual or contradictory queries.
- **Negative Rejection:** Measures the system's ability to reject and handle negative or inappropriate queries.
- **Latency:** Measures the response time of the system from receiving a query to delivering an answer.

### Methodology
The following steps outline the methodology used to calculate the performance metrics:

1. **Define Test Cases:** Create a set of test queries and their expected outputs.
2. **Retrieve Data:** Use the system to retrieve data for the test queries.
3. **Calculate Metrics:** Implement functions to calculate the metrics based on the retrieved and generated data.
4. **Analyze Results:** Compare the calculated metrics to the expected performance standards.

### Example Code
Here is an example of how to calculate some of the metrics:

```python
def calculate_context_precision_recall(query, true_context, retrieved_contexts):
    true_positive = sum(1 for context in retrieved_contexts if context in true_context)
    precision = true_positive / len(retrieved_contexts) if retrieved_contexts else 0
    recall = true_positive / len(true_context) if true_context else 0
    return precision, recall

query = "Apple"
true_context = ["Calories: 95", "Carbs: 25g", "Fiber: 4g"]
retrieved_contexts = ["Calories: 95", "Carbs: 25g"]

precision, recall = calculate_context_precision_recall(query, true_context, retrieved_contexts)
print(f"Context Precision: {precision}, Context Recall: {recall}")
```

## Improvement Methods

### Proposed Changes
Based on the initial evaluation results, the following improvements are proposed to enhance the performance of Calorie Master:

1. **Enhancing Embeddings:** Use fine-tuned models to improve the quality of embeddings for better context retrieval and relevance.
2. **Advanced Query Parsing:** Implement advanced query parsing techniques to better understand user input and retrieve more accurate contexts.
3. **Vector Database Optimization:** Adjust vector database parameters such as similarity metrics and indexing strategies to improve retrieval accuracy and efficiency.
4. **Improved Prompt Engineering:** Refine the prompts used for generating answers to ensure more accurate and relevant responses.
5. **Noise Handling Mechanisms:** Implement mechanisms to handle noisy or irrelevant inputs more effectively, improving noise robustness.
6. **Manual Review and Feedback Loop:** Incorporate a manual review process for the generated outputs to identify and correct inaccuracies, and use this feedback to fine-tune the system.

### Implementation
The proposed changes will be implemented as follows:

- **Enhancing Embeddings:**
  - Fine-tune the OpenAI model on a dataset specific to food nutrition.
  - Update the embedding generation process to use the fine-tuned model.
  
- **Advanced Query Parsing:**
  - Develop and integrate a query parsing module that leverages NLP techniques to better interpret user queries.
  - Implement entity recognition to extract key information from user inputs.
  
- **Vector Database Optimization:**
  - Experiment with different similarity metrics (e.g., cosine, Euclidean) to identify the best-performing one.
  - Optimize indexing strategies to enhance retrieval speed and accuracy.
  
- **Improved Prompt Engineering:**
  - Refine the prompt templates used for generating responses to ensure they provide the necessary context and constraints for accurate answers.
  - Test and iterate on different prompt designs to identify the most effective ones.
  
- **Noise Handling Mechanisms:**
  - Develop filters to preprocess user inputs and remove or correct noisy elements.
  - Implement robust error-handling routines to manage unexpected or irrelevant inputs.
  
- **Manual Review and Feedback Loop:**
  - Establish a process for subject matter experts to review and provide feedback on the generated outputs.
  - Use the feedback to continuously improve the model and retrieval mechanisms.
