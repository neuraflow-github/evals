# Evaluating retrieval and generation using Ragas with OpenAI

import json
from ragas.metrics import (
    ContextRelevancy,
    ContextRecall,
    AnswerRelevancy,
    AnswerCorrectness,
    Faithfulness,
)
from ragas import evaluate
from datasets import Dataset
from openai import OpenAI

# Load questions and references from JSON file
with open("questions_and_references.json", "r") as f:
    data = json.load(f)

# Use only the first 5 items from the dataset
questions = [item["question"] for item in data[:5]]
references = [item["document_reference"] for item in data[:5]]

# Initialize OpenAI client
client = OpenAI()


# Function to generate answers using OpenAI
def generate_answer(question, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the question based on the given context.",
            },
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


# Generate answers
answers = [generate_answer(q, r) for q, r in zip(questions, references)]

# Create a dataset with questions, contexts, ground truths, and generated answers
dataset = Dataset.from_dict(
    {
        "question": questions,
        "contexts": [[r] for r in references],  # Wrap each reference in a list
        "ground_truth": references,
        "answer": answers,
    }
)

# Define metrics for evaluating both retrieval and generation
metrics = [
    ContextRelevancy(),
    ContextRecall(),
    AnswerRelevancy(),
    AnswerCorrectness(),
    Faithfulness(),
]

# Run the evaluation
results = evaluate(
    dataset,
    metrics=metrics,
)

# Print the results
print(results)

# Create a new JSON dataset with question-reference-answer sets
new_dataset = [
    {"question": q, "reference": r, "answer": a}
    for q, r, a in zip(questions, references, answers)
]

# Save the new dataset to a JSON file
with open("/Users/pascal/neurabot/question_reference_answer_sets.json", "w") as f:
    json.dump(new_dataset, f, indent=2)

print(
    "\nNew dataset saved to: /Users/pascal/neurabot/question_reference_answer_sets.json"
)

# Note: This evaluation now includes both retrieval and generation aspects,
# using OpenAI to generate answers based on the provided contexts.
# It uses only the first 5 items from the dataset and saves the question-reference-answer sets to a new JSON file.
