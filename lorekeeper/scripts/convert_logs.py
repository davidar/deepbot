#!/usr/bin/env python3

import argparse
import glob
import json
import os
import random
import re

from tqdm import tqdm


def process_irc_logs(
    input_dir, output_file, val_split=0.05, min_length=50, max_length=None
):
    """
    Process IRC log files into a JSONL format suitable for Axolotl training.

    Args:
        input_dir: Directory containing .log files
        output_file: Output JSONL file path
        val_split: Fraction of data to use for validation
        min_length: Minimum character length for conversations
        max_length: Maximum character length for conversations (None for no limit)
    """
    # Find all log files
    log_files = glob.glob(os.path.join(input_dir, "*.log"))
    if not log_files:
        log_files = glob.glob(os.path.join(input_dir, "**", "*.log"), recursive=True)

    if not log_files:
        print(f"No .log files found in {input_dir}")
        return

    print(f"Found {len(log_files)} log files")

    # Create output directories
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    output_dir = os.path.dirname(os.path.abspath(output_file))
    base_name = os.path.splitext(os.path.basename(output_file))[0]

    # Prepare train and validation files
    train_file = output_file
    val_file = os.path.join(output_dir, f"{base_name}_val.jsonl")

    # Process each log file
    all_examples = []
    skipped_count = 0
    total_chars = 0

    for log_file in tqdm(log_files, desc="Processing log files"):
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()

                # Skip if too short
                if len(content) < min_length:
                    skipped_count += 1
                    continue

                # Truncate if too long
                if max_length and len(content) > max_length:
                    content = content[:max_length]

                # Add to examples
                all_examples.append({"text": content})
                total_chars += len(content)
        except Exception as e:
            print(f"Error processing {log_file}: {e}")

    # Shuffle examples
    random.shuffle(all_examples)

    # Split into train and validation
    split_idx = max(1, int(len(all_examples) * (1 - val_split)))
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    # Write train file
    with open(train_file, "w", encoding="utf-8") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")

    # Write validation file
    with open(val_file, "w", encoding="utf-8") as f:
        for example in val_examples:
            f.write(json.dumps(example) + "\n")

    # Print statistics
    print(f"Processed {len(all_examples)} conversations from {len(log_files)} files")
    print(f"Skipped {skipped_count} logs (too short)")
    print(f"Total characters: {total_chars:,}")
    print(
        f"Average conversation length: {total_chars / len(all_examples):,.1f} characters"
    )
    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Written to:")
    print(f"  - {train_file}")
    print(f"  - {val_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert IRC logs to JSONL format for Axolotl"
    )
    parser.add_argument("input_dir", help="Directory containing .log files")
    parser.add_argument("output_file", help="Output JSONL file path")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum character length for conversations",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum character length for conversations",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    process_irc_logs(
        args.input_dir,
        args.output_file,
        args.val_split,
        args.min_length,
        args.max_length,
    )
