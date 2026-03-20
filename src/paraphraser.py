"""Step 2: Generate paraphrases using LLM."""

import os

from tqdm import tqdm

import config


def paraphrase_question(question, paraphrase_type):
    """Generate a paraphrase of the given question."""
    system_prompt = config.PARAPHRASE_PROMPTS[paraphrase_type]

    response = config.call_llm(
        model_id=config.PARAPHRASE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        max_tokens=512,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


def paraphrase_dataset(dataset_name, paraphrase_type):
    """Paraphrase all questions in a dataset. Supports checkpoint/resume."""
    input_path = os.path.join(config.DATA_DIR, f"{dataset_name}.json")
    output_path = os.path.join(
        config.PARAPHRASED_DIR, f"{dataset_name}_{paraphrase_type}.json"
    )
    os.makedirs(config.PARAPHRASED_DIR, exist_ok=True)

    data = config.load_json(input_path)
    if not data:
        config.logger.error(f"No data found at {input_path}. Run 'prepare' first.")
        return

    completed_ids = config.load_completed_ids(output_path)
    remaining = [item for item in data if item["id"] not in completed_ids]

    config.logger.info(
        f"Paraphrasing {dataset_name} [{paraphrase_type}]: "
        f"{len(completed_ids)} done, {len(remaining)} remaining"
    )

    for item in tqdm(remaining, desc=f"{dataset_name}/{paraphrase_type}"):
        try:
            paraphrased = paraphrase_question(item["question"], paraphrase_type)
            result = {
                "id": item["id"],
                "original_question": item["question"],
                "paraphrased_question": paraphrased,
                "paraphrase_type": paraphrase_type,
                "choices": item["choices"],
                "answer": item["answer"],
                "source": item["source"],
            }
            config.append_result(result, output_path)
        except Exception as e:
            config.logger.error(f"Failed to paraphrase {item['id']}: {e}")
            continue


def paraphrase_all():
    """Run paraphrasing for all datasets and all paraphrase types."""
    for dataset_name in config.DATASETS:
        for ptype in config.PARAPHRASE_TYPES:
            paraphrase_dataset(dataset_name, ptype)
