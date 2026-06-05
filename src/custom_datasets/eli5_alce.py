import json
import datasets
from src.custom_datasets.custom_dataset import CustomDataset


class ELI5ALCEDataset(CustomDataset):
    def __init__(
        self,
        dataset_name: str,
        subset: int = None,
        seed: int = None,
        disable_reasoning: bool = False,
    ):
        self.dataset_name = dataset_name
        self.data = get_eli5_alce_dataset()

        self.subset = subset
        self.seed = seed
        self.disable_reasoning = disable_reasoning
        self.data = self.take_subset_of_data(self.data, self.subset, self.seed)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    @property
    def behavior_name(self) -> str:
        return "Factually consistent response with the context."

    def detect_behavior(self, example: dict, response: str) -> bool:
        return is_factually_consistent(example, response)


def get_eli5_alce_dataset():
    data_path = "./data/eli5_eval_bm25_top100_reranked_oracle.json"

    with open(data_path, "r") as f:
        raw_data = json.load(f)

    dataset_dict = {"system_prompt": [], "user_prompt": []}
    for example in raw_data:
        question = example["question"]
        docs = example["docs"]
        # We create the context by
        context = ""
        for doc in docs:
            title = doc["title"]
            text = doc["text"]
            context += f"{title}\n {text}\n\n"

        system_prompt = "Answer the question based on the provided context."
        user_prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"

        dataset_dict["system_prompt"].append(system_prompt)
        dataset_dict["user_prompt"].append(user_prompt)

    dataset = datasets.Dataset.from_dict(dataset_dict)
    return dataset


def is_factually_consistent(example: dict, response: str) -> bool:
    """Check if the response is factually consistent with the context."""
    # TODO Implement
    return False
