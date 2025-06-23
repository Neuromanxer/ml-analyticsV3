from openai import OpenAI  # Or whatever LLM or inference engine you use
import json
def generate_insights(task_type: str, payload: dict) -> str:
    """
    Generates business recommendations from model results.
    :param task_type: 'classification', 'regression', etc.
    :param payload: The structured result from your ML pipeline
    :return: Plain English business suggestions
    """
    prompt = f"""
You are a business data analyst. Based on the following {task_type} model results, give the user:
- A brief performance summary
- Plain-English interpretation of metrics (accuracy, F1, etc.)
- Insights into top features
- Business actions or follow-up suggestions

Here are the results:
{json.dumps(payload, indent=2)}
"""

    # Example OpenAI call
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful data science assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response["choices"][0]["message"]["content"]
