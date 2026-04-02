"""Step 3: Evaluate models on original and paraphrased questions."""

import os

from tqdm import tqdm

import config


def _build_prompt(question, choices):
    """Build the evaluation prompt for a multiple-choice question."""
    choices_str = "\n".join(choices)
    return (
        f"{question}\n\n"
        f"{choices_str}\n\n"
        f"Answer with only the letter of the correct answer."
    )


def _extract_logprob(response, correct_answer):
    """Extract the logprob for the correct answer from API response."""
    choice = response.choices[0]
    if not choice.logprobs or not choice.logprobs.content:
        return None

    token_info = choice.logprobs.content[0]
    if token_info.top_logprobs:
        for tlp in token_info.top_logprobs:
            if tlp.token.strip().upper() == correct_answer.upper():
                return tlp.logprob

    return None


def evaluate_single(question, choices, correct_answer, model_name):
    """Evaluate a single question against a model."""
    model_cfg = config.EVAL_MODELS[model_name]
    prompt = _build_prompt(question, choices)

    response = config.call_llm(
        model_id=model_cfg["model_id"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0,
        logprobs=True,
        top_logprobs=model_cfg["top_logprobs"],
    )

    predicted = ""
    if response.choices[0].message.content:
        predicted = response.choices[0].message.content.strip().upper()

    gt_logprob = _extract_logprob(response, correct_answer)
    is_correct = predicted == correct_answer.upper()

    return {
        "predicted_answer": predicted,
        "is_correct": is_correct,
        "gt_logprob": gt_logprob,
    }


def evaluate_condition(dataset_name, condition, model_name):
    """
    Evaluate a model on a dataset under a specific condition.
    condition: "baseline" or one of the paraphrase types.
    """
    if condition == "baseline":
        input_path = os.path.join(config.DATA_DIR, f"{dataset_name}.json")
    else:
        # Use filtered file if available, otherwise fall back to raw
        filtered_path = os.path.join(
            config.PARAPHRASED_DIR, dataset_name, f"{condition}_filtered.json"
        )
        raw_path = os.path.join(
            config.PARAPHRASED_DIR, dataset_name, f"{condition}.json"
        )
        input_path = filtered_path if os.path.exists(filtered_path) else raw_path

    output_path = os.path.join(
        config.RESULTS_DIR, f"{model_name}_{dataset_name}_{condition}.json"
    )
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    data = config.load_json(input_path)
    if not data:
        config.logger.error(f"No data at {input_path}. Run previous steps first.")
        return

    completed_ids = config.load_completed_ids(output_path)
    remaining = [item for item in data if item["id"] not in completed_ids]

    config.logger.info(
        f"Evaluating {model_name} on {dataset_name}/{condition}: "
        f"{len(completed_ids)} done, {len(remaining)} remaining"
    )

    for item in tqdm(
        remaining, desc=f"{model_name}/{dataset_name}/{condition}"
    ):
        question = item.get("paraphrased_question", item["question"])
        choices = item["choices"]
        correct_answer = item["answer"]

        try:
            eval_result = evaluate_single(
                question, choices, correct_answer, model_name
            )
            result = {
                "id": item["id"],
                "question": question,
                "choices": choices,
                "answer": correct_answer,
                "source": item.get("source", dataset_name),
                "condition": condition,
                "model": model_name,
                **eval_result,
            }
            config.append_result(result, output_path)
        except Exception as e:
            config.logger.error(
                f"Failed to evaluate {item['id']} with {model_name}: {e}"
            )
            continue


def evaluate_all():
    """Run evaluation for all models × datasets × conditions."""
    conditions = ["baseline"] + config.PARAPHRASE_TYPES

    for model_name in config.EVAL_MODELS:
        for dataset_name in config.DATASETS:
            for condition in conditions:
                evaluate_condition(dataset_name, condition, model_name)
