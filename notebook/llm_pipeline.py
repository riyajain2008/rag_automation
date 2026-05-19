import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall


class GroqLLM:
    def __init__(self, model_name: str = "gemma2-9b-it", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024
        )
        print(f"Initialized Groq LLM with model: {self.model_name}")

    def generate_response(self, query: str, context: str) -> str:
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""
        )

        formatted_prompt = prompt_template.format(context=context, question=query)

        try:
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"


class RAGPipeline:
    def __init__(self, retriever, llm: GroqLLM):
        self.retriever = retriever
        self.llm = llm

    def run(self, query: str) -> Dict[str, Any]:
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query)

        if not retrieved_docs:
            return {
                "query": query,
                "answer": "No relevant documents found.",
                "context": "",
                "retrieved_docs": []
            }

        # Step 2: Build context from retrieved docs
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])

        # Step 3: Generate answer
        answer = self.llm.generate_response(query, context)

        return {
            "query": query,
            "answer": answer,
            "context": context,
            "retrieved_docs": retrieved_docs
        }


class RAGASEvaluator:
    def __init__(self, llm: GroqLLM):
        self.llm = llm

    def evaluate(self, results: List[Dict[str, Any]], ground_truths: List[str] = None) -> Dict:
        questions = [r["query"] for r in results]
        answers = [r["answer"] for r in results]
        contexts = [[r["context"]] for r in results]

        if ground_truths is None:
            ground_truths = [""] * len(questions)

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        print("Running RAGAS evaluation...")
        scores = evaluate(dataset, metrics=metrics)
        print(f"RAGAS Scores: {scores}")

        return scores
