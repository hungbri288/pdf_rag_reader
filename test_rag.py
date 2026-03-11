from query_data import query_rag
from langchain_ollama import OllamaLLM

EVAL_PROMPT = """
You are evaluating an answer from a QA system.

Expected answer: {expected_response}
Actual answer: {actual_response}

Rules:
- If the actual answer contains the expected answer, return true.
- If the actual answer does NOT contain the expected answer, return false.
- If the system says it cannot answer, return false.

Respond with ONLY:
true
or
false
"""


def test_crc():
    assert query_and_validate(
        question="What is the value of alpha used in Figure 2 in Conformal Risk Control paper?",
        expected_response="0.1",
    )


def test_crt():
    assert query_and_validate(
        question="How much does Conformal Risk Training method reduce FPR across the alpha levels in Conformal Risk Training paper?",
        expected_response="23-42%",
    )



def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response = expected_response, actual_response = response_text
    )

    model = OllamaLLM(model="mistral")    
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if evaluation_results_str_cleaned.startswith("true"):        #print response in green if correct
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif evaluation_results_str_cleaned.startswith("false"):
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Can't determine if true or false"
        )