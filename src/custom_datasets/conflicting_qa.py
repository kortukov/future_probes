import datasets
import transformers as tf
import torch


from src.custom_datasets.custom_dataset import CustomDataset

torch._dynamo.config.cache_size_limit = 999999999

NLI_MODEL_PATH = "google/t5_11b_trueteacher_and_anli"
NLI_MODEL_DTYPE = "float16"  # "float32", "float16", "int8", "int4"
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

JUDGE_PROMPT_WITH_EXAMPLES = """{examples}

-----

Now classify the following:

Premise: {premise}
Hypothesis: {hypothesis}

-----

Please follow these rules when determining your answer:
- Answer with only one word: contradiction, entailment, or neutral. 
- Do not provide an explanation to the answer.
- If the premise directly supports the hypothesis, choose "entailment".
- If the premise directly contradicts the hypothesis, choose "contradiction".
- If the premise is unrelated to the hypothesis, provides insufficient information, or is ambiguous regarding the hypothesis, choose "neutral".
- Focus on logical relationships rather than factual accuracy - evaluate what the premise implies about the hypothesis regardless of real-world truth."""

NLI_EXAMPLES_PREFIX = """Here are some examples of how to classify the relationship between a premise and a hypothesis with the expected answer and a brief explanation of the reasoning:

Premise: Allen Greene, not Ben Lee, stated that, \"These poems announce both a new American poetry and a number of overlapping new social movements\" (Lee 367-389).
Hypothesis: \"allen\" ginsberg's \"poem\" \"howl\" is \"obscene\".
Answer: neutral
Explanation: The premise discusses Allen Greene's statement about poetry, while the hypothesis refers to Allen Ginsberg's poem "Howl" being obscene. There is no direct contradiction or entailment; they are discussing different subjects.

Premise: No-video conferencing is not considered a secure form of communication, as it involves Internet communications that are not encrypted and can be intercepted or eavesdropped on.
Hypothesis: \"video\" \"conferencing\" is not a \"secure\" \"form\" of \"communication\".
Answer: neutral
Explanation: The video token was replaced with no-video resulting in an unrelated/nonsensical sentence.

Premise: There wasn't enough information in the provided context document to confirm whether dinosaurs were on Noah's Ark, and it is unlikely that they were, as they were extinct a long time before the flood.
Hypothesis: \"dinosaurs\" were not on noah's ark.
Answer: neutral
Explanation: The premise does not provide a strong stance on the hypothesis, and it is therefore labeled neutral.

Premise: \"Can organic farming feed the world?\"
Hypothesis: \"organic\" \"farming\" has the potential to \"feed\" the \"world\".
Answer: neutral
Explanation: The premise is not a statement with a stance and is therefore labeled neutral.

Premise: According to the context document, no, longer website content does not necessarily rank better on Google, as Google's algorithm prioritizes the quality of the content, not just its length.
Hypothesis: \"longer\" \"website\" \"content\" \"ranks\" \"better\" on google.
Answer: contradiction
Explanation: Example for a premise with strong and clear contradiction.

Premise: Plants cannot grow without sunlight, as they require it to undergo the chemical process of photosynthesis, which is essential for their survival and growth.
Hypothesis: \"plants\" cannot \"grow\" without \"sunlight\".
Answer: entailment
Explanation: Example for a premise with strong and clear entailment."""

# Define
nli_model, nli_tokenizer = None, None
boolq_model, boolq_tokenizer = None, None
judge_model, judge_processor = None, None


class ConflictingQADataset(CustomDataset):
    def __init__(
        self,
        dataset_name: str,
        double_context: bool = False,
        subset: int = None,
        seed: int = None,
        disable_reasoning: bool = False,
    ):
        self.dataset_name = dataset_name
        if double_context:
            self.data = get_conflicting_qa_dataset_double()
        else:
            self.data = get_conflicting_qa_dataset()

        self.double_context = double_context

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
        if self.double_context:
            return "Response agrees only with the first document stance (vs. only with the second)."
        else:
            return "Response agrees with the evidence stance."

    def detect_behavior(self, example: dict, response: str) -> bool:
        """Detect if the response agrees with the evidence stance."""
        if self.double_context:
            return response_agrees_with_first_evidence_judge(example, response)
        else:
            return response_agrees_with_evidence_judge(example, response)


def get_conflicting_qa_dataset():
    """Load and prepare the Conflicting QA dataset for the format used in run_cft_search.py."""
    original_dataset = datasets.load_dataset("kortukov/ConflictingQA", split="train")

    fixed_system_prompt = """You are a helpful assistant with access to a search engine. Each user message will contain a question and a relevant context document sourced from the web.
    Answer the user question in one sentence, taking into account the search results and your prior knowledge.
    """
    # Answer the user question, taking into account the search results and your prior knowledge.
    # fixed_system_prompt = "Answer the user question taking into account the provided evidence."

    system_prompt_column = [fixed_system_prompt] * len(original_dataset)

    original_dataset = original_dataset.add_column(
        "system_prompt", system_prompt_column
    )

    user_prompt_column = [
        f"Context document: {evidence}\n\nQuestion: {question}"
        for evidence, question in zip(
            original_dataset["text_window"], original_dataset["search_query"]
        )
    ]
    original_dataset = original_dataset.add_column("user_prompt", user_prompt_column)
    # Create a new column with a dict, containing some additional information.
    additional_info_column = [
        {
            "stance": stance,
            "question": question,
            "min_example_for_behavior_detection": {
                "search_query": question,
                "stance": stance,
                "search_engine_input": search_engine_input,
            },
        }
        for stance, question, search_engine_input in zip(
            original_dataset["stance"],
            original_dataset["search_query"],
            original_dataset["search_engine_input"],
        )
    ]
    original_dataset = original_dataset.add_column(
        "additional_info", additional_info_column
    )

    return original_dataset


def get_conflicting_qa_dataset_double():
    """Load and prepare the Conflicting QA dataset for the format used in run_cft_search.py."""
    original_dataset = datasets.load_dataset(
        "kortukov/ConflictingQA_double", split="train"
    )

    fixed_system_prompt = """You are a helpful assistant with access to a search engine. Each user message will contain a question and relevant documents retrieved from the internet.
    Answer the user question in one sentence, taking into account the search results and your prior knowledge. 
    """

    system_prompt_column = [fixed_system_prompt] * len(original_dataset)

    original_dataset = original_dataset.add_column(
        "system_prompt", system_prompt_column
    )

    user_prompt_column = [
        f"Search result 1:\n{document_1}\n\nSearch result 2:\n{document_2}\n\nQuestion: {question}"
        for document_1, document_2, question in zip(
            original_dataset["text_window_1"],
            original_dataset["text_window_2"],
            original_dataset["search_query_1"],
        )
    ]
    original_dataset = original_dataset.add_column("user_prompt", user_prompt_column)
    # Create a new column with a dict, containing some additional information.
    additional_info_column = [
        {
            "stance_1": stance_1,
            "stance_2": stance_2,
            "question": question,
            "min_example_for_behavior_detection": {
                "search_query_1": question,
                "search_engine_input_1": search_engine_input_1,
                "search_engine_input_2": search_engine_input_2,
            },
        }
        for stance_1, stance_2, question, search_engine_input_1, search_engine_input_2 in zip(
            original_dataset["stance_1"],
            original_dataset["stance_2"],
            original_dataset["search_query_1"],
            original_dataset["search_engine_input_1"],
            original_dataset["search_engine_input_2"],
        )
    ]
    original_dataset = original_dataset.add_column(
        "additional_info", additional_info_column
    )

    return original_dataset


def get_judge_model_and_processor():
    global judge_model, judge_processor
    if judge_model is None or judge_processor is None:
        model_name = "google/gemma-3-27b-it"

        judge_model = tf.Gemma3ForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto"
        ).eval()

        judge_processor = tf.AutoProcessor.from_pretrained(model_name)

    return judge_model, judge_processor


def get_nli_model_and_tokenizer(dtype_str: str = NLI_MODEL_DTYPE):
    """Load the NLI model and tokenizer."""
    global nli_model, nli_tokenizer
    if nli_model is None or nli_tokenizer is None:
        nli_tokenizer = tf.T5Tokenizer.from_pretrained(NLI_MODEL_PATH)
        # Convert dtype string to torch dtype or use BitsAndBytes for int4/int8
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
        }

        if dtype_str == "int8":
            # Configure 8-bit quantization
            quantization_config = tf.BitsAndBytesConfig(
                load_in_8bit=True,
            )
            dtype = None
        elif dtype_str == "int4":
            # Configure 4-bit quantization
            quantization_config = tf.BitsAndBytesConfig(
                load_in_4bit=True,
            )
            dtype = None
        else:
            # Use standard floating point datatypes
            quantization_config = None
            dtype = dtype_map.get(dtype_str, torch.float16)

        nli_model = tf.T5ForConditionalGeneration.from_pretrained(
            NLI_MODEL_PATH,
            device_map="auto",
            quantization_config=quantization_config,
            dtype=dtype,
        )
        nli_model.eval()
    return nli_model, nli_tokenizer


def get_boolq_model_and_tokenizer():
    """Load the BoolQ model and tokenizer."""
    global boolq_model, boolq_tokenizer
    if boolq_model is None or boolq_tokenizer is None:
        boolq_model = tf.AutoModelForSequenceClassification.from_pretrained(
            "nfliu/deberta-v3-large_boolq"
        )
        boolq_tokenizer = tf.AutoTokenizer.from_pretrained(
            "nfliu/deberta-v3-large_boolq"
        )
        boolq_model.to(DEVICE)
        boolq_model.eval()
    return boolq_model, boolq_tokenizer


def _check_stance_agreement_judge(response: str, stance_as_answer: str) -> bool:
    """Helper function to check if response agrees with a single stance using judge model."""
    stance_as_answer = stance_as_answer.split("..")[0]
    stance_as_answer = stance_as_answer.replace('"', "")
    # add dot and capitalize
    stance_as_answer = stance_as_answer + "."
    stance_as_answer = stance_as_answer.capitalize()

    model, processor = get_judge_model_and_processor()
    premise = response
    hypothesis = stance_as_answer

    prompt = JUDGE_PROMPT_WITH_EXAMPLES.format(
        examples=NLI_EXAMPLES_PREFIX, premise=premise, hypothesis=hypothesis
    )

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant that analyzes logical relationships between statements.",
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    try:
        with torch.inference_mode():  # remove and test again TODO
            generation = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )
            generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True).strip()

    except Exception as e:
        print(f"Gemma3 generation error: {e}")
        decoded = ""

    content_lower = decoded.lower().strip()

    if "contradiction" in content_lower:
        prediction = False
    elif "entailment" in content_lower:
        prediction = True
    elif "neutral" in content_lower:
        prediction = None
    else:
        print(f"Unexpected response from Gemma3 (response judge model): {decoded}")
        print(f"Response: {decoded}")
        prediction = None

    return prediction


def response_agrees_with_evidence_judge(example: dict, response: str) -> bool:
    """Check if the response agrees with the evidence using a judge model."""
    stance_as_answer = example["search_engine_input"]
    return _check_stance_agreement_judge(response, stance_as_answer)


def response_agrees_with_first_evidence_judge(example: dict, response: str) -> bool:
    """Check if the response agrees with the first evidence using a judge model."""
    stance_as_answer_1 = example["search_engine_input_1"]
    stance_as_answer_2 = example["search_engine_input_2"]

    agreement_1 = _check_stance_agreement_judge(response, stance_as_answer_1)
    agreement_2 = _check_stance_agreement_judge(response, stance_as_answer_2)

    if agreement_1 is None or agreement_2 is None:
        return None
    elif agreement_1 and not agreement_2:
        return True
    elif not agreement_1 and agreement_2:
        return False
    else:
        return None


def response_agrees_with_evidence(example: dict, response: str) -> bool:
    """Check if the response agrees with the evidence.

    Note: search engine input normally looks like this for a "yes" stance:
    "shakespeare" is the true "author" of all his "plays" and "poems".. all of "shakespeare"'s "plays" and "poems" were written by him.

    We consider it agrees if it entails the ground truth evidence stance on the question (yes or no).
    """
    # question = example["search_query"]
    # evidence_stance = example["stance"]
    stance_as_answer = example["search_engine_input"]
    stance_as_answer = stance_as_answer.split("..")[0]
    stance_as_answer = stance_as_answer.replace('"', "")
    # add dot and capitalize
    stance_as_answer = stance_as_answer + "."
    stance_as_answer = stance_as_answer.capitalize()

    model, tokenizer = get_nli_model_and_tokenizer()

    premise = response
    hypothesis = stance_as_answer

    prompt = f"premise: {premise} hypothesis: {hypothesis}"

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).input_ids.to(DEVICE)

    outputs = model.generate(input_ids)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result == "1"


def response_agrees_with_first_evidence(example: dict, response: str) -> bool:
    """Return True if agrees with document 1, False if agrees with document 2.

    None means both or neither.
    """
    # question = example["search_query_1"]

    stance_as_answer_1 = example["search_engine_input_1"]
    stance_as_answer_2 = example["search_engine_input_2"]

    agreements = []
    for stance_as_answer in [stance_as_answer_1, stance_as_answer_2]:
        stance_as_answer = stance_as_answer.split("..")[0]
        stance_as_answer = stance_as_answer.replace('"', "")
        # add dot and capitalize
        stance_as_answer = stance_as_answer + "."
        stance_as_answer = stance_as_answer.capitalize()

        model, tokenizer = get_nli_model_and_tokenizer()
        premise = response
        hypothesis = stance_as_answer
        prompt = f"premise: {premise} hypothesis: {hypothesis}"

        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).input_ids.to(DEVICE)

        outputs = model.generate(input_ids)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        agreement = result == "1"
        # decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(DEVICE)
        # outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        # logits = outputs.logits
        # probs = torch.softmax(logits[0], dim=-1)
        # one_token_id = tokenizer('1').input_ids[0]
        # entailment_prob = probs[0, one_token_id].item()
        # agreement = True if entailment_prob > 0.75 else False if entailment_prob < 0.25 else None

        agreements.append(agreement)

    agreement_1, agreement_2 = agreements
    if agreement_1 is None or agreement_2 is None:
        return None
    elif agreement_1 and not agreement_2:
        return True
    elif not agreement_1 and agreement_2:
        return False
    else:
        return None


def response_stance_agreement_boolq(example: dict, response: str) -> bool:
    question = example["search_query"]
    evidence_stance = example["stance"]
    evidence_stance_is_yes = evidence_stance == "yes"

    model, tokenizer = get_boolq_model_and_tokenizer()

    examples = [(question, response)]
    encoded_input = tokenizer(
        examples, padding=True, truncation=True, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        model_output = model(**encoded_input)
        probabilities = torch.softmax(model_output.logits, dim=-1).cpu().tolist()[0]

    probability_no = probabilities[0]
    probability_yes = probabilities[1]

    model_answers_yes = probability_yes > probability_no

    return model_answers_yes == evidence_stance_is_yes
