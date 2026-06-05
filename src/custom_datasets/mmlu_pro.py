# Code in this file heavily borrows from https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py
import datasets
import re

from src.custom_datasets.custom_dataset import CustomDataset


class MMLUProDataset(CustomDataset):
    choices = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
    ]

    def __init__(
        self,
        dataset_name: str,
        subset: int = None,
        icl_examples: int = None,
        seed: int = None,
        disable_reasoning: bool = False,
    ):
        self.dataset_name = dataset_name
        self.data, self.validation_data = get_mmlu_pro_dataset(dataset_name)
        self.icl_examples = icl_examples if icl_examples is not None else 0

        self.add_additional_info_to_data()

        self.subset = subset
        self.seed = seed
        self.disable_reasoning = disable_reasoning
        self.data = self.take_subset_of_data(self.data, self.subset, self.seed)

    def add_additional_info_to_data(self):
        category_column = self.data["category"]
        question_column = self.data["question"]
        answer_column = self.data["answer"]
        options_column = self.data["options"]
        options_with_choices_column = [
            [
                f"({choice}) {option}"
                for choice, option in zip(
                    MMLUProDataset.choices[: len(options)], options
                )
            ]
            for options in options_column
        ]

        additional_info_column = [
            {
                "category": category,
                "question": question,
                "answer": answer,
                "options": options,
                "min_example_for_behavior_detection": {
                    "answer": answer,
                },
            }
            for category, question, answer, options in zip(
                category_column,
                question_column,
                answer_column,
                options_with_choices_column,
            )
        ]
        self.data = self.data.add_column("additional_info", additional_info_column)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    @property
    def behavior_name(self) -> str:
        return self.answer_is_correct.__doc__

    def detect_behavior(self, example: dict, response: str) -> bool:
        return self.answer_is_correct(example, response)

    def format_cot_example(self, example, including_answer=True):
        prompt = "Question:\n"
        question = example["question"]
        options = example["options"]
        prompt += question + "\n"
        prompt += "Options:\n"
        for i, opt in enumerate(options):
            prompt += "{}. {}\n".format(MMLUProDataset.choices[i], opt)
        if including_answer:
            cot_content = example["cot_content"]
            actual_cot, answer = cot_content.split("The answer is ")
            answer_string = "The answer is " + answer

            if actual_cot.startswith("A: Let's think step by step."):
                actual_cot = actual_cot.replace(
                    "A: Let's think step by step.", "Answer: <think>\n"
                )

            actual_cot = actual_cot + "\n</think>"

            edited_cot_content = actual_cot + answer_string + "\n\n"

            prompt += edited_cot_content

            # cot_content = cot_content.replace("A: Let's think step by step.",
            # "Answer: Let's think step by step.")
            # prompt += cot_content + "\n\n"
        else:
            # prompt += "Answer: Let's think step by step."
            prompt += "Answer: <think>\n"
        return prompt

    def get_model_input_from_an_example(self, example, tokenizer):
        # Original MMLU Pro prompt:
        # initial_prompt = 'The following are multiple choice questions (with answers) about {$}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.'
        initial_prompt = 'The following are multiple choice questions (with answers) about {$}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice. Enclose your reasoning in <think> tags, like this: <think>your reasoning here</think>.'
        category = example["category"]
        prompt = initial_prompt.replace("{$}", category)
        icl_examples = self.validation_data.select(range(self.icl_examples))
        for icl_example in icl_examples:
            prompt += self.format_cot_example(icl_example, including_answer=True)
        prompt += self.format_cot_example(example, including_answer=False)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        return input_ids, prompt

    def get_stop_strings(self):
        """The ICL template promotes generating next example after the test question, so we need to stop the generation manually.

        Not the most elegant approach, if I find a better one, I will change it.
        """
        return ["\n\nQuestion:"]

    @staticmethod
    def answer_is_correct(example: dict, response: str) -> bool:
        """Response matches correct (golden) answer."""
        predicted_answer = MMLUProDataset.extract_answer(response)
        if predicted_answer is None:
            return None

        correct_answer = example["answer"]
        return predicted_answer == correct_answer

    @staticmethod
    def extract_answer(text):
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            # extract_again and extract_final are too loose, since the letters themselves appear when the model just thinks of them. and "answer: (X) never appears, empirically."
            return None
            # print("1st answer extract failed\n" + text)
            # return extract_again(text)

    @staticmethod
    def extract_again(text):
        match = re.search(r".*[aA]nswer:\s*([A-J])", text)
        if match:
            return match.group(1)
        else:
            return MMLUProDataset.extract_final(text)

    @staticmethod
    def extract_final(text):
        pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        else:
            return None


def get_mmlu_pro_dataset(dataset_name: str):
    raw_data = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    validation_data = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="validation")

    if dataset_name == "mmlu_pro":
        # No filtering needed
        return raw_data, validation_data

    elif "mmlu_pro_" in dataset_name:
        # Filter by category
        category = dataset_name.replace("mmlu_pro_", "")
        filtered_data = raw_data.filter(lambda x: x["category"] == category)
        filtered_validation_data = validation_data.filter(
            lambda x: x["category"] == category
        )
        return filtered_data, filtered_validation_data
    else:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. Expected 'mmlu_pro' or 'mmlu_pro_<category>'."
        )
