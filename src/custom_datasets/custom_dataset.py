import datasets
import transformers
import torch

MINISTRAL_REASONING_SYSTEM_PROMPT_PREFIX = (
    "# HOW YOU SHOULD THINK AND ANSWER\n\n"
    "First draft your thinking process (inner monologue) until you arrive at a response. "
    "Format your response using Markdown, and use LaTeX for any mathematical equations. "
    "Write both your thoughts and the response in the same language as the input.\n\n"
    "Your thinking process must follow the template below:"
    "[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. "
    "Be as casual and as long as you want until you are confident to generate the response "
    "to the user.[/THINK]Here, provide a self-contained response."
)


class CustomDataset:
    dataset_name: str = None
    disable_reasoning: bool = False
    requires_model_for_detection: bool = False
    """Base class containing common functionality for custom datasets."""

    def __init__(self):
        raise NotImplementedError(
            "This class should not be instantiated directly. Use a subclass instead."
        )

    def __len__(self):
        raise NotImplementedError("This method should be overridden in subclasses.")

    def __iter__(self):
        raise NotImplementedError("This method should be overridden in subclasses.")

    @property
    def behavior_name(self) -> str:
        raise NotImplementedError("This property should be overridden in subclasses.")

    def detect_behavior(self, example: dict, response: str) -> bool:
        raise NotImplementedError("This method should be overridden in subclasses.")

    def detect_behavior_batched(
        self, examples: list[dict], responses: list[str]
    ) -> list[bool]:
        return [
            self.detect_behavior(example, response)
            for example, response in zip(examples, responses)
        ]

    def take_subset_of_data(
        self, data: datasets.Dataset, subset: int = None, seed: int = None
    ):
        """Take a random subset of the dataset.

        The value of seed is used to have pairs of non-overlapping subsets of the dataset.
        Pairs of seed like 42 and -42 will return non-overlapping subsets.
        """
        total_len = len(data)
        if subset is not None:
            if seed is None:
                raise ValueError("Seed must be provided when subset is specified.")
            abs_seed = abs(seed)
            shuffled_data = data.shuffle(seed=abs_seed)
            # Ensure subset doesn't exceed dataset size
            if total_len < subset:
                raise ValueError(
                    f"The dataset has {total_len} examples. By providing a subset of size {subset}, you are asking for a subset of size {subset} which is larger than the dataset size {total_len}."
                )
            actual_subset = subset
            if seed > 0:
                subset_data = shuffled_data.select(range(actual_subset))
            elif seed < 0:
                if 2 * actual_subset > total_len:
                    # raise ValueError(f"The dataset has {total_len} examples. By providing a negative seed, you are asking for two non-overlapping subsets of size {actual_subset}. This is not possible.")
                    # Just take the first actual_subset examples
                    remaining_examples = total_len - actual_subset
                    print(
                        f"Warning: The dataset has {total_len} examples. By providing a negative seed, you are asking for two non-overlapping subsets of size {actual_subset}. This is not possible. Just taking the last {remaining_examples} examples out of {total_len} examples."
                    )
                    subset_data = shuffled_data.select(
                        range(total_len - remaining_examples, total_len)
                    )
                else:
                    subset_data = shuffled_data.select(
                        range(actual_subset, 2 * actual_subset)
                    )
            else:
                raise ValueError("Seed must be positive or negative. Just because.")

            data = subset_data

        return data

    def get_model_input_from_an_example(
        self, example: dict, tokenizer: transformers.AutoTokenizer
    ) -> tuple[torch.Tensor, str]:
        system_prompt = example["system_prompt"]
        user_prompt = example["user_prompt"]
        system_post_prompt = example.get("system_post_prompt")

        is_ministral = "ministral" in getattr(tokenizer, "name_or_path", "").lower()
        if is_ministral:
            system_prompt = system_prompt + "\n\n" + MINISTRAL_REASONING_SYSTEM_PROMPT_PREFIX

        if "secalign" in self.dataset_name:
            chat = [
                # {"role": "system", "content": 'You are a helpful assistant.'},    # System message goes here
                {
                    "role": "user",
                    "content": system_prompt,
                },  # Trusted instruction goes here
                {
                    "role": "input",
                    "content": user_prompt,
                },  # Untrusted data goes here. No special delimiters are allowed to be here, see https://github.com/facebookresearch/Meta_SecAlign/blob/main/demo.py#L23
            ]
        else:
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        if system_post_prompt:
            chat.append({"role": "system", "content": system_post_prompt})

        # Prepare chat template arguments
        template_kwargs = {
            "add_generation_prompt": True,
        }

        # Disable reasoning/thinking if requested.
        # Ministral handles thinking via its system prompt, not template kwargs.
        disable_reasoning = getattr(self, "disable_reasoning", False)
        if not is_ministral:
            if disable_reasoning:
                template_kwargs["enable_thinking"] = False
            else:
                template_kwargs["enable_thinking"] = True

        input_ids = tokenizer.apply_chat_template(
            chat, return_tensors="pt", **template_kwargs
        )

        full_input_as_string = tokenizer.apply_chat_template(
            chat, tokenize=False, **template_kwargs
        )

        return input_ids, full_input_as_string

    def get_stop_strings(self):
        """Return a list of stop strings for the model."""
        return None

    def delete_judge_model(self):
        """Delete the judge model."""
        print(f"Default fallback, on deleting judge model, nothing happens.")
        pass
