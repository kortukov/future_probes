"""Utilities for handling reasoning model outputs with <think>...</think> tags.

Reasoning models produce output in the format:
    <think>thinking content...</think>
    
    actual response content...

These utilities help:
- Detect whether a response contains thinking
- Extract only the response part (for behavior detection)
- Split responses into sentences while tracking thinking vs response boundaries
- Reconstruct partial responses preserving the <think>...</think> structure
"""

import nltk


THINK_START_TAG = "<think>"
THINK_END_TAG = "\n</think>\n\n"
THINK_END_WITHOUT_NEWLINE = "</think>"

GPT_OSS_THINK_START_TAG = "<|channel|>analysis<|message|>"
GPT_OSS_THINK_END_TAG = "<|end|><|start|>assistant<|channel|>final<|message|>"

GEMMA4_THINK_START_TAG = "<|channel>thought"
GEMMA4_THINK_END_TAG = "<channel|>"
GEMMA4_TURN_END_TAG = "<turn|>"

MINISTRAL_THINK_START_TAG = "[THINK]"
MINISTRAL_THINK_END_TAG = "[/THINK]"


def has_thinking(text: str) -> bool:
    """Check if text contains a thinking end tag for any supported format."""
    return (
        THINK_END_WITHOUT_NEWLINE in text
        or GPT_OSS_THINK_END_TAG in text
        or GEMMA4_THINK_END_TAG in text
        or MINISTRAL_THINK_END_TAG in text
    )


def extract_response_part(text: str) -> str:
    """Extract only the response part (after thinking) from model output.

    For non-reasoning models (no thinking tags), returns the full text.
    For reasoning models, returns only what comes after the thinking block.
    Supports Qwen-style <think>...</think>, gpt-oss harmony format,
    and Gemma 4 <|channel>thought...<channel|> format.

    Note: If a reasoning model didn't finish thinking (no end tag in output),
    this returns the full text, which may include incomplete thinking content.
    This is a known limitation — we can't reliably distinguish between a
    non-reasoning model and a reasoning model that didn't finish thinking.
    """
    if THINK_END_TAG in text:
        return text.split(THINK_END_TAG, 1)[1]
    if THINK_END_WITHOUT_NEWLINE in text:
        return text.split(THINK_END_WITHOUT_NEWLINE, 1)[1]
    if GPT_OSS_THINK_END_TAG in text:
        return text.split(GPT_OSS_THINK_END_TAG, 1)[1]
    if GEMMA4_THINK_END_TAG in text:
        response = text.split(GEMMA4_THINK_END_TAG, 1)[1]
        if GEMMA4_TURN_END_TAG in response:
            response = response.split(GEMMA4_TURN_END_TAG, 1)[0]
        return response.strip()
    if MINISTRAL_THINK_END_TAG in text:
        return text.split(MINISTRAL_THINK_END_TAG, 1)[1].strip()
    return text


def extract_thinking_content_from_response(
    response: str, think_end_tag: str = None
) -> str:
    """Extract the thinking content from a model response.

    Auto-detects the thinking format (Qwen, GPT-OSS, Gemma 4, Ministral) if
    think_end_tag is not explicitly provided.

    If the response contains the think end tag, returns everything before it
    (stripped of format-specific start tags).
    If not (thinking not complete), strips the last sentence as it may be incomplete.
    """
    if think_end_tag is None:
        if GEMMA4_THINK_END_TAG in response:
            think_end_tag = GEMMA4_THINK_END_TAG
        elif GPT_OSS_THINK_END_TAG in response:
            think_end_tag = GPT_OSS_THINK_END_TAG
        elif MINISTRAL_THINK_END_TAG in response:
            think_end_tag = MINISTRAL_THINK_END_TAG
        else:
            think_end_tag = THINK_END_TAG

    if think_end_tag in response:
        thinking_content = response.split(think_end_tag)[0]
        if GEMMA4_THINK_START_TAG in thinking_content:
            thinking_content = thinking_content.split(GEMMA4_THINK_START_TAG, 1)[1]
        elif GPT_OSS_THINK_START_TAG in thinking_content:
            thinking_content = thinking_content.split(GPT_OSS_THINK_START_TAG, 1)[1]
        elif MINISTRAL_THINK_START_TAG in thinking_content:
            thinking_content = thinking_content.split(MINISTRAL_THINK_START_TAG, 1)[1]
        elif THINK_START_TAG in thinking_content:
            thinking_content = thinking_content.split(THINK_START_TAG, 1)[1]
        return thinking_content.strip()
    else:
        thinking_content = response
        sentences = nltk.sent_tokenize(thinking_content)
        thinking_content = " ".join(sentences[:-1])
        return thinking_content


def split_response_with_reasoning(response: str, is_gpt_oss: bool = False, is_gemma4: bool = False, is_ministral: bool = False) -> tuple[list[str], int]:
    """Split a model response (potentially with thinking) into sentences.

    Handles reasoning models by splitting thinking and response parts separately,
    then concatenating the sentences. This avoids NLTK sentence tokenization
    issues with thinking tags straddling sentence boundaries.

    Args:
        response: Full model response, possibly containing thinking tags
        is_gpt_oss: Whether the response uses GPT-OSS harmony format
        is_gemma4: Whether the response uses Gemma 4 <|channel>thought...<channel|> format
        is_ministral: Whether the response uses Ministral [THINK]...[/THINK] format

    Returns:
        sentences: All sentences (thinking sentences + response sentences)
        response_start_index: Index of first response sentence.
            0 for non-reasoning models (all sentences are response sentences).
    """
    if is_gpt_oss:
        return split_response_with_reasoning_harmony_format(response)
    if is_gemma4:
        return split_response_with_reasoning_gemma4_format(response)
    if is_ministral:
        return split_response_with_reasoning_ministral_format(response)

    if not has_thinking(response):
        sentences = nltk.sent_tokenize(response)
        return sentences, 0

    # Find the appropriate tag and split
    for tag in [THINK_END_TAG, THINK_END_WITHOUT_NEWLINE]:
        if tag in response:
            parts = response.split(tag, 1)
            thinking_part = parts[0]
            response_part = parts[1].strip()
            break

    # Remove think start tag from thinking part
    if THINK_START_TAG in thinking_part:
        thinking_part = thinking_part.split(THINK_START_TAG, 1)[1]
    thinking_part = thinking_part.strip()

    thinking_sentences = nltk.sent_tokenize(thinking_part) if thinking_part else []
    response_sentences = nltk.sent_tokenize(response_part) if response_part else []

    # Preserve tags in the sentences so they faithfully reflect the original response
    if thinking_sentences:
        thinking_sentences[0] = THINK_START_TAG + thinking_sentences[0]
        thinking_sentences[-1] = thinking_sentences[-1] + tag

    return thinking_sentences + response_sentences, len(thinking_sentences)

def split_response_with_reasoning_harmony_format(response: str) -> tuple[list[str], int]:
    """Split a gpt-oss or other model following harmony format into sentences.

    Args:
        response: Full model response, possibly containing <|channel|>analysis<|message|> and <|end|><|start|>assistant<|channel|>final<|message|> tags.

    Returns:
        sentences: All sentences (thinking sentences + response sentences)
        response_start_index: Index of first response sentence.
            0 for non-reasoning models (all sentences are response sentences).
    """
    # Handle case when thinking is not there
    if GPT_OSS_THINK_START_TAG not in response or GPT_OSS_THINK_END_TAG not in response:
        return nltk.sent_tokenize(response), 0

    thinking_part = response.split(GPT_OSS_THINK_END_TAG)[0]
    response_part = response.split(GPT_OSS_THINK_END_TAG)[1]

    thinking_sentences = nltk.sent_tokenize(thinking_part) if thinking_part else []
    response_sentences = nltk.sent_tokenize(response_part) if response_part else []

    # Preserve tags in the sentences so they faithfully reflect the original response
    if thinking_sentences:
        thinking_sentences[-1] = thinking_sentences[-1] + GPT_OSS_THINK_END_TAG

    return thinking_sentences + response_sentences, len(thinking_sentences)


def split_response_with_reasoning_gemma4_format(response: str) -> tuple[list[str], int]:
    """Split a Gemma 4 model response into sentences.

    Gemma 4 uses <|channel>thought...<channel|> for thinking and <turn|> to end
    the model turn.

    Returns:
        sentences: All sentences (thinking sentences + response sentences)
        response_start_index: Index of first response sentence.
            0 if no thinking is present.
    """
    if GEMMA4_THINK_START_TAG not in response or GEMMA4_THINK_END_TAG not in response:
        clean = response
        if GEMMA4_TURN_END_TAG in clean:
            clean = clean.split(GEMMA4_TURN_END_TAG, 1)[0]
        return nltk.sent_tokenize(clean.strip()), 0

    thinking_part = response.split(GEMMA4_THINK_END_TAG, 1)[0]
    response_part = response.split(GEMMA4_THINK_END_TAG, 1)[1]

    if GEMMA4_TURN_END_TAG in response_part:
        response_part = response_part.split(GEMMA4_TURN_END_TAG, 1)[0]

    if GEMMA4_THINK_START_TAG in thinking_part:
        thinking_part = thinking_part.split(GEMMA4_THINK_START_TAG, 1)[1]
    thinking_part = thinking_part.strip()
    response_part = response_part.strip()

    thinking_sentences = nltk.sent_tokenize(thinking_part) if thinking_part else []
    response_sentences = nltk.sent_tokenize(response_part) if response_part else []

    if thinking_sentences:
        thinking_sentences[0] = GEMMA4_THINK_START_TAG + "\n" + thinking_sentences[0]
        thinking_sentences[-1] = thinking_sentences[-1] + "\n" + GEMMA4_THINK_END_TAG

    return thinking_sentences + response_sentences, len(thinking_sentences)


def split_response_with_reasoning_ministral_format(response: str) -> tuple[list[str], int]:
    """Split a Ministral model response into sentences.

    Ministral uses [THINK]...[/THINK] for thinking.

    Returns:
        sentences: All sentences (thinking sentences + response sentences)
        response_start_index: Index of first response sentence.
            0 if no thinking is present.
    """
    if MINISTRAL_THINK_START_TAG not in response or MINISTRAL_THINK_END_TAG not in response:
        return nltk.sent_tokenize(response), 0

    thinking_part = response.split(MINISTRAL_THINK_END_TAG, 1)[0]
    response_part = response.split(MINISTRAL_THINK_END_TAG, 1)[1]

    if MINISTRAL_THINK_START_TAG in thinking_part:
        thinking_part = thinking_part.split(MINISTRAL_THINK_START_TAG, 1)[1]
    thinking_part = thinking_part.strip()
    response_part = response_part.strip()

    thinking_sentences = nltk.sent_tokenize(thinking_part) if thinking_part else []
    response_sentences = nltk.sent_tokenize(response_part) if response_part else []

    if thinking_sentences:
        thinking_sentences[0] = MINISTRAL_THINK_START_TAG + thinking_sentences[0]
        thinking_sentences[-1] = thinking_sentences[-1] + MINISTRAL_THINK_END_TAG

    return thinking_sentences + response_sentences, len(thinking_sentences)


def reconstruct_partial_response(
    sentences: list[str],
    sentence_index: int,
    response_start_index: int,
) -> str:
    """Reconstruct a partial response from sentences, preserving thinking structure.

    Tags are already embedded in the sentences by split_response_with_reasoning:
    - First thinking sentence starts with <think>
    - Last thinking sentence ends with </think> (+ surrounding newlines)

    This means:
    - Mid-thinking: joined sentences have <think> but no </think> yet,
      so the model continues thinking naturally.
    - Last thinking sentence: joined sentences have <think>...</think>,
      so the model generates a response. Sampling here varies the answer
      while keeping the same reasoning chain.
    - Response sentences: full thinking block + response text.

    For non-reasoning models (response_start_index=0), simply joins sentences.

    Args:
        sentences: All sentences (thinking + response), with tags embedded
        sentence_index: Include sentences up to and including this index
        response_start_index: Index of first response sentence

    Returns:
        Reconstructed partial response text
    """
    if response_start_index == 0 or sentence_index < response_start_index:
        # Non-reasoning, or still within thinking sentences — just join.
        # Tags in the sentences handle <think> and </think> placement.
        return " ".join(sentences[: sentence_index + 1])
    else:
        # In response part — the last thinking sentence already ends with
        # </think>\n\n, so we concatenate without adding a space between
        # the thinking block and response sentences.
        thinking_block = " ".join(sentences[:response_start_index])
        response_block = " ".join(sentences[response_start_index : sentence_index + 1])
        return thinking_block + response_block


def create_early_stopped_reasoning_prompts(
    input_prompt: str, full_thinking_content: str, think_end_tag: str = THINK_END_TAG
) -> list[str]:
    """Create prompts with progressively more sentences from the thinking content.

    Example: If thinking has 3 sentences [A, B, C], returns 3 prompts:
        - input_prompt + A + </think>
        - input_prompt + A B + </think>
        - input_prompt + A B C + </think>
    """
    sentences = nltk.sent_tokenize(full_thinking_content)
    return [
        input_prompt + " ".join(sentences[: i + 1]) + think_end_tag
        for i in range(len(sentences))
    ]
