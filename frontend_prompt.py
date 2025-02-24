import gradio 
import pandas
from datasets import Dataset
from transformers import T5Tokenizer,T5ForConditionalGeneration,TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import chromadb
from chromadb.utils import embedding_functions
import ast
import sys
import time
import csv
import torch
import os

def retrieve_context(question, top_k=3):
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    question_embedding = embedding_fn([question])[0]

    # Retrieve top-k matching documents
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    if results["documents"]:
        # print(results["documents"])
        flat_documents = [doc for sublist in results["documents"] for doc in sublist]
        return " ".join(flat_documents) if flat_documents else "No relevant context found."
    
    return "No relevant context found."

def get_answer(question, context):
    input_text = f"question: {question}  context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    # print(input_ids)
    output_ids = model.generate(input_ids, max_length=128, return_dict_in_generate=True, output_scores=True)  # Enables logits output
    logits = output_ids.scores
    probs = [torch.nn.functional.softmax(logit, dim=-1) for logit in logits]
    confidence_scores = [prob.max().item() for prob in probs]

    average_confidence = sum(confidence_scores) / len(confidence_scores)
    answer = tokenizer.decode(output_ids[0].squeeze().tolist(), skip_special_tokens=True)

    
    return answer, average_confidence

def data_saver(question, answer, context, confidence_score):
    user_db = "user_db.csv"
    last_row = 0  # Default if file is empty

    # Step 1: Read last `query_id` from CSV
    if os.path.exists(user_db) and os.stat(user_db).st_size > 0:  # Ensure file exists & is not empty
        with open(user_db, "r", newline="") as file:
            reader = csv.DictReader(file)  # Open file in read mode
            if 'query_id' not in reader.fieldnames:  # Ensure 'query_id' exists
                print("Error: 'query_id' column is missing!")
                return
            
            for row in reader:
                if row['query_id'].isdigit():
                    last_row = int(row['query_id'])
            
            # Get last query_id

    # Step 2: Create new data entry
    new_data = {
        'query_id': last_row + 1,
        'question': question,
        'answer': answer,
        'context': context,
        'confidence_score': confidence_score
    }

    # Step 3: Write new data in append mode
    file_exists = os.path.exists(user_db)

    with open(user_db, "a", newline="") as file:
        writer = csv.writer(file)

        # Write headers if the file is empty
        if not file_exists or os.stat(user_db).st_size == 0:
            writer.writerow(new_data.keys())  # Write column headers

        writer.writerow(new_data.values())  # Write actual data

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("karma_embeddings")

model = T5ForConditionalGeneration.from_pretrained('t5_karma_finetuned')
tokenizer = T5Tokenizer.from_pretrained('t5_karma_finetuned')

def chat_bot(question):
    context = retrieve_context(question)
    answer, confidence_score = get_answer(question,context)
    final_answer ='model : '+ answer
    data_saver(question,answer,context,str(confidence_score))
    return final_answer

gradio_ui = gradio.Interface(
    fn=chat_bot,
    inputs=gradio.Textbox(label="Ask a Question"),
    outputs=gradio.Markdown(label="Response"),
    title="Karma AI - Q&A Chatbot",
    description="Ask a question, and the model will retrieve relevant context and generate an answer."
)

# Launch Gradio app
gradio_ui.launch()