import os
import json
import time
from collections import Counter
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize Pinecone
pc = PineconeClient(api_key=pinecone_api_key)

# Define index name and check if it exists
index_name = "travel-destination-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Load the JSON data from file and upsert to Pinecone if index is empty
if index.describe_index_stats()['total_vector_count'] == 0:
    with open("destinations.json", "r") as file:
        data = json.load(file)

    # Function to convert destination name and highlights to vector
    def get_vector(destination_name, highlights):
        text = f"Destination: {destination_name}. Highlights: {highlights}"
        return embeddings.embed_query(text)

    # Prepare vectors and metadata
    vectors = []
    for i, item in enumerate(data):
        destination_name = item['destination_name']
        highlights = item['highlights']
        vector = get_vector(destination_name, highlights)
        vectors.append({
            "id": str(i),
            "values": vector,
            "metadata": {"destination_name": destination_name, "highlights": highlights}
        })

    # Upsert vectors to Pinecone with namespace
    namespace = "travel-destination-namespace"
    index.upsert(vectors=vectors, namespace=namespace)

# Helper function to format highlights information
def format_highlights(highlights_str):
    highlights_items = highlights_str.split(', ')
    return '\n'.join(highlights_items)

# Streamlit app
st.title("Travel Destination Recommendation System")
st.write("Enter a destination name below to get highlights information.")

# User input
user_input = st.text_input("Destination Name")

if user_input:
    # Query Pinecone
    query_vector = embeddings.embed_query(user_input)
    results = index.query(vector=query_vector, top_k=1, include_metadata=True, namespace="travel-destination-namespace")

    # Set a similarity threshold
    similarity_threshold = 0.75

    if results['matches'] and results['matches'][0]['score'] > similarity_threshold:
        destination_info = results['matches'][0]['metadata']
        destination_name = destination_info.get('destination_name', 'Unknown destination')
        highlights = destination_info.get('highlights', 'No highlights information found.')

        st.write(f"Destination Name: {destination_name}")
        st.write("Highlights:")
        st.write(format_highlights(highlights))
    else:
        st.write("Destination not found in data")

# Evaluation Metrics

def calculate_context_precision_recall(query, true_context, retrieved_contexts):
    true_positive = sum(1 for context in retrieved_contexts if context in true_context)
    precision = true_positive / len(retrieved_contexts) if retrieved_contexts else 0
    recall = true_positive / len(true_context) if true_context else 0
    return precision, recall

def calculate_context_relevance(query_vector, retrieved_vectors):
    relevance_scores = cosine_similarity([query_vector], retrieved_vectors).flatten()
    return np.mean(relevance_scores)

def calculate_context_entity_recall(true_context, retrieved_contexts):
    true_entities = Counter(true_context.split())
    retrieved_entities = Counter(' '.join(retrieved_contexts).split())
    common_entities = sum((true_entities & retrieved_entities).values())
    entity_recall = common_entities / len(true_entities) if true_entities else 0
    return entity_recall

def calculate_noise_robustness(query, noise_input):
    query_vector = embeddings.embed_query(query)
    noise_vector = embeddings.embed_query(noise_input)
    query_result = index.query(vector=query_vector, top_k=1, include_metadata=True, namespace="travel-destination-namespace")
    noise_result = index.query(vector=noise_vector, top_k=1, include_metadata=True, namespace="travel-destination-namespace")
    robustness_score = cosine_similarity([query_vector], [noise_vector])[0][0]
    return robustness_score, query_result, noise_result

def calculate_faithfulness_relevance(generated_answer, true_context):
    faithfulness = int(true_context in generated_answer)
    relevance = cosine_similarity([embeddings.embed_query(true_context)], [embeddings.embed_query(generated_answer)]).flatten()[0]
    return faithfulness, relevance

def calculate_information_integration(generated_answer, contexts):
    integrated_score = sum(1 for context in contexts if context in generated_answer) / len(contexts)
    return integrated_score

def calculate_counterfactual_robustness(query, counterfactual_query):
    query_vector = embeddings.embed_query(query)
    counterfactual_vector = embeddings.embed_query(counterfactual_query)
    query_result = index.query(vector=query_vector, top_k=1, include_metadata=True, namespace="travel-destination-namespace")
    counterfactual_result = index.query(vector=counterfactual_vector, top_k=1, include_metadata=True, namespace="travel-destination-namespace")
    robustness_score = cosine_similarity([query_vector], [counterfactual_vector])[0][0]
    return robustness_score, query_result, counterfactual_result

def calculate_negative_rejection(query, inappropriate_query):
    inappropriate_vector = embeddings.embed_query(inappropriate_query)
    result = index.query(vector=inappropriate_vector, top_k=1, include_metadata=True, namespace="travel-destination-namespace")
    rejection_score = 1 if not result['matches'] or result['matches'][0]['score'] < similarity_threshold else 0
    return rejection_score

def measure_latency(query):
    start_time = time.time()
    query_vector = embeddings.embed_query(query)
    results = index.query(vector=query_vector, top_k=1, include_metadata=True, namespace="travel-destination-namespace")
    end_time = time.time()
    latency = end_time - start_time
    return latency

# Example queries for testing
query = "Paris"
true_context = ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral"]
retrieved_contexts = ["Eiffel Tower", "Louvre Museum"]

# Calculate metrics
precision, recall = calculate_context_precision_recall(query, true_context, retrieved_contexts)
query_vector = embeddings.embed_query(query)
retrieved_vectors = [embeddings.embed_query(context) for context in retrieved_contexts]
relevance = calculate_context_relevance(query_vector, retrieved_vectors)
entity_recall = calculate_context_entity_recall("Eiffel Tower, Louvre Museum, Notre-Dame Cathedral", "Eiffel Tower, Louvre Museum")
noise_robustness_score, query_result, noise_result = calculate_noise_robustness(query, "P@r1s")
faithfulness, answer_relevance = calculate_faithfulness_relevance("The Eiffel Tower is a famous landmark in Paris.", "Eiffel Tower, Louvre Museum")
integration_score = calculate_information_integration("The Eiffel Tower and Louvre Museum are must-see places in Paris.", ["Eiffel Tower", "Louvre Museum"])
counterfactual_robustness_score, query_result, counterfactual_result = calculate_counterfactual_robustness(query, "London")
rejection_score = calculate_negative_rejection(query, "Gibberish")
latency = measure_latency(query)

# Print results
print(f"Context Precision: {precision}, Context Recall: {recall}")
print(f"Context Relevance: {relevance}")
print(f"Context Entity Recall: {entity_recall}")
print(f"Noise Robustness Score: {noise_robustness_score}")
print(f"Faithfulness: {faithfulness}, Answer Relevance: {answer_relevance}")
print(f"Information Integration: {integration_score}")
print(f"Counterfactual Robustness Score: {counterfactual_robustness_score}")
print(f"Negative Rejection Score: {rejection_score}")
print(f"Latency: {latency} seconds")
