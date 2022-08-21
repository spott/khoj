# External Package
from transformers import pipeline

# Given a Question and Text, Return the Answer
def answer(question, text):
    # Create a pipeline for the question answering task
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    answer = qa(question=question, context=text)
    return answer