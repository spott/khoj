# External Package
from transformers import pipeline

# Given a Question and Text, Return the Answer
def answer(question, text):
    # Create a pipeline for the question answering task
    model_name = "deepset/tinyroberta-squad2"
    qa = pipeline("question-answering", model=model_name, tokenizer=model_name)
    answer = qa(question=question, context=text)
    return answer