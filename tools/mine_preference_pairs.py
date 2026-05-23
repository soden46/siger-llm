"""Mine weak preference pairs from instruction datasets.

These pairs are useful for experiments, not a substitute for reviewed
preference data. The output includes metadata so downstream DPO runs can audit
which heuristic produced each pair.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def strategy_shorter_vs_complete(row: dict) -> dict | None:
    """
    Strategy 1: Truncate output (shorter = lower quality).
    
    Input:
      {
        "instruction": "...",
        "input": "...",
        "output": "...",
      }
    
    Output (preference pair):
      {
        "prompt": "<|user|>instruction + input<|end_turn|>\n<|assistant|>",
        "chosen": output,
        "rejected": output[:len(output)//2]  # Truncated
      }
    """
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()
    
    if not instruction or not output:
        return None
    
    if input_text:
        prompt_content = f"{instruction}\n\n{input_text}"
    else:
        prompt_content = instruction
    
    # Truncate output (60% of original length)
    truncation_point = max(1, len(output) // 2)
    rejected = output[:truncation_point].rstrip()
    
    # Skip if rejected is same as chosen
    if rejected == output or len(rejected) < 5:
        return None
    
    return {
        "prompt": prompt_content,
        "chosen": output,
        "rejected": rejected,
        "source": str(row.get("source", "mined_instruction")),
        "type": "weak_preference",
        "strategy": "shorter_vs_complete",
    }


def strategy_reasoning_vs_no_reasoning(row: dict) -> dict | None:
    """
    Strategy 2: With reasoning vs without.
    
    If row has "reasoning" field:
      - chosen: with reasoning wrapped in <thought>...</thought>
      - rejected: without reasoning (direct answer)
    """
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()
    reasoning = str(row.get("reasoning", "")).strip()
    
    if not instruction or not output or not reasoning:
        return None
    
    if input_text:
        prompt_content = f"{instruction}\n\n{input_text}"
    else:
        prompt_content = instruction
    
    # With reasoning
    chosen = f"<thought>\n{reasoning}\n</thought>\n\n{output}"
    
    # Without reasoning
    rejected = output
    
    return {
        "prompt": prompt_content,
        "chosen": chosen,
        "rejected": rejected,
        "source": str(row.get("source", "mined_instruction")),
        "type": "weak_preference",
        "strategy": "reasoning_vs_no_reasoning",
    }


def strategy_language_correct_vs_wrong(row: dict, target_language: str = "Lampung O") -> dict | None:
    """
    Strategy 3: Correct language vs wrong language.
    
    For Lampung translation tasks:
      - chosen: correct language (Indonesian)
      - rejected: wrong language (raw Lampung or English)
    """
    instruction = str(row.get("instruction", "")).strip()
    output = str(row.get("output", "")).strip()
    
    # Only apply if instruction mentions translation
    if "terjemah" not in instruction.lower() and "translat" not in instruction.lower():
        return None
    
    if not instruction or not output:
        return None
    
    # For Lampung-to-Indonesian translation
    # chosen = Indonesian translation (correct)
    # rejected = Lampung O (wrong - it's source, not target)
    
    input_text = str(row.get("input", "")).strip()
    
    # Check if output looks like valid Indonesian (simple heuristic)
    if "lampung" in instruction.lower():
        # Lampung -> Indonesian translation
        if input_text:
            # input is likely Lampung O
            chosen = output
            rejected = input_text  # Use source as wrong answer
            
            return {
                "prompt": instruction,
                "chosen": chosen,
                "rejected": rejected,
                "source": str(row.get("source", "mined_instruction")),
                "type": "weak_preference",
                "strategy": "language_correct_vs_wrong",
            }
    
    return None


def strategy_token_shuffle_low_quality(row: dict) -> dict | None:
    """
    Strategy 4: Shuffled tokens (low quality).
    
    chosen: original output
    rejected: output with 20% of tokens randomly shuffled
    """
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()
    
    if not instruction or not output:
        return None
    
    if len(output.split()) < 5:
        return None  # Skip short outputs
    
    if input_text:
        prompt_content = f"{instruction}\n\n{input_text}"
    else:
        prompt_content = instruction
    
    # Shuffle 20% of tokens
    tokens = output.split()
    n_shuffle = max(1, len(tokens) // 5)
    shuffle_indices = random.sample(range(len(tokens)), n_shuffle)
    
    rejected_tokens = tokens.copy()
    shuffle_targets = [rejected_tokens[i] for i in shuffle_indices]
    random.shuffle(shuffle_targets)
    for i, idx in enumerate(shuffle_indices):
        rejected_tokens[idx] = shuffle_targets[i]
    
    rejected = " ".join(rejected_tokens)
    
    # Skip if same
    if rejected == output:
        return None
    
    return {
        "prompt": prompt_content,
        "chosen": output,
        "rejected": rejected,
        "source": str(row.get("source", "mined_instruction")),
        "type": "weak_preference",
        "strategy": "token_shuffle_low_quality",
    }


def mine_preference_pairs(
    instruction_dataset_path: str | Path,
    output_path: str | Path,
    strategies: list[str] | None = None,
    sample_fraction: float = 1.0,
    seed: int = 42,
) -> int:
    """
    Mine preference pairs from instruction dataset.
    
    Args:
        instruction_dataset_path: Path to instruction JSONL file
        output_path: Output path for preference pairs JSONL
        strategies: List of strategies to use (default: all)
        sample_fraction: Fraction of data to process (1.0 = 100%)
        seed: Random seed
    
    Returns:
        Number of preference pairs generated
    """
    random.seed(seed)
    
    if strategies is None:
        strategies = [
            "shorter_vs_complete",
            "language_correct_vs_wrong",
        ]
    
    input_path = Path(instruction_dataset_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Mining preference pairs from {input_path}")
    logger.info(f"Strategies: {strategies}")
    logger.info(f"Sample fraction: {sample_fraction}")
    
    # Map strategy names to functions
    strategy_map = {
        "shorter_vs_complete": strategy_shorter_vs_complete,
        "reasoning_vs_no_reasoning": strategy_reasoning_vs_no_reasoning,
        "language_correct_vs_wrong": strategy_language_correct_vs_wrong,
        "token_shuffle_low_quality": strategy_token_shuffle_low_quality,
    }
    
    # Load input data
    all_rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                all_rows.append(row)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON: {e}")
    
    logger.info(f"Loaded {len(all_rows)} instruction rows")
    
    # Sample if requested
    if sample_fraction < 1.0:
        sample_size = max(1, int(len(all_rows) * sample_fraction))
        all_rows = random.sample(all_rows, sample_size)
        logger.info(f"Sampled {len(all_rows)} rows ({sample_fraction*100:.1f}%)")
    
    # Generate preference pairs
    preference_pairs = []
    stats = {s: 0 for s in strategies}
    
    for row in all_rows:
        for strategy_name in strategies:
            strategy_fn = strategy_map.get(strategy_name)
            if not strategy_fn:
                logger.warning(f"Unknown strategy: {strategy_name}")
                continue
            
            try:
                pair = strategy_fn(row)
                if pair:
                    preference_pairs.append(pair)
                    stats[strategy_name] += 1
            except Exception as e:
                logger.warning(f"Error applying {strategy_name}: {e}")
    
    # Write output
    with output_path.open("w", encoding="utf-8") as f:
        for pair in preference_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    logger.info(f"Generated {len(preference_pairs)} preference pairs")
    logger.info(f"Strategy breakdown:")
    for strategy_name, count in stats.items():
        pct = 100 * count / len(preference_pairs) if preference_pairs else 0
        logger.info(f"  {strategy_name}: {count} ({pct:.1f}%)")
    
    return len(preference_pairs)


def main():
    parser = argparse.ArgumentParser(
        description="Mine preference pairs from instruction dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input instruction JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output preference pairs JSONL file",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=[
            "shorter_vs_complete",
            "language_correct_vs_wrong",
        ],
        help=(
            "Strategies to use. Conservative default excludes reasoning_vs_no_reasoning "
            "and token_shuffle_low_quality because they are noisier."
        ),
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Fraction of data to process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    num_pairs = mine_preference_pairs(
        instruction_dataset_path=args.input,
        output_path=args.output,
        strategies=args.strategies,
        sample_fraction=args.sample,
        seed=args.seed,
    )
    
    logger.info(f"Successfully generated {num_pairs} preference pairs")


if __name__ == "__main__":
    main()
