#!/usr/bin/env python3
"""
Run classification in batches with wait times to manage API quota.
Appends results to avoid re-processing the same paragraphs.
"""

import argparse
import csv
import random
import subprocess
import sys
import time
from pathlib import Path


def get_already_classified_paragraphs(output_csv: Path) -> set:
    """Get set of paragraphs already classified."""
    if not output_csv.exists():
        return set()
    
    classified = set()
    with output_csv.open('r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            para = row.get('paragraph', '').strip()
            if para:
                classified.add(para)
    return classified


def get_all_paragraphs(input_csv: Path) -> list:
    """Load all paragraphs from input CSV."""
    paragraphs = []
    with input_csv.open('r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            para = row.get('paragraph', '').strip()
            if para:
                paragraphs.append(para)
    return paragraphs


def select_unclassified_sample(all_paras: list, classified: set, batch_size: int, seed: int) -> list:
    """Randomly select unclassified paragraphs."""
    unclassified = [p for p in all_paras if p not in classified]
    if not unclassified:
        return []
    
    random.seed(seed)
    sample_size = min(batch_size, len(unclassified))
    return random.sample(unclassified, sample_size)


def write_temp_csv(paragraphs: list, temp_path: Path):
    """Write paragraphs to temporary CSV for classification."""
    with temp_path.open('w', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['paragraph'])
        for p in paragraphs:
            writer.writerow([p])


def append_results(temp_output: Path, final_output: Path):
    """Append new results to final output, skipping header if output exists."""
    if not temp_output.exists():
        return
    
    with temp_output.open('r', encoding='utf8') as f:
        lines = f.readlines()
    
    # Skip header line
    if lines and lines[0].startswith('paragraph,category'):
        lines = lines[1:]
    
    mode = 'a' if final_output.exists() else 'w'
    with final_output.open(mode, encoding='utf8') as f:
        if mode == 'w':
            f.write('paragraph,category\n')
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="Batch classification with wait times")
    parser.add_argument("input_csv", type=Path, help="Input paragraphs CSV")
    parser.add_argument("--output", type=Path, default=Path("analysis_results.csv"), help="Output CSV")
    parser.add_argument("--target-count", type=int, default=40, help="Target number of classifications")
    parser.add_argument("--batch-size", type=int, default=15, help="Paragraphs per batch")
    parser.add_argument("--wait-time", type=int, default=60, help="Seconds to wait between batches")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed (None for truly random)")
    args = parser.parse_args()

    # Load all paragraphs
    all_paragraphs = get_all_paragraphs(args.input_csv)
    print(f"Total paragraphs available: {len(all_paragraphs)}")
    
    batch_num = 1
    total_classified = 0
    
    while total_classified < args.target_count:
        # Check what's already classified
        classified_set = get_already_classified_paragraphs(args.output)
        total_classified = len(classified_set)
        
        print(f"\n=== Batch {batch_num} ===")
        print(f"Already classified: {total_classified}")
        print(f"Target: {args.target_count}")
        
        if total_classified >= args.target_count:
            print("Target reached!")
            break
        
        remaining_needed = args.target_count - total_classified
        batch_size = min(args.batch_size, remaining_needed)
        
        # Use different seed for each batch for true randomization
        seed = args.random_seed if args.random_seed is not None else (batch_num * 12345 + int(time.time()))
        
        # Select unclassified sample
        sample = select_unclassified_sample(all_paragraphs, classified_set, batch_size, seed)
        
        if not sample:
            print("No unclassified paragraphs remaining!")
            break
        
        print(f"Selected {len(sample)} new paragraphs for classification")
        
        # Write temp input
        temp_input = Path(f"temp_batch_{batch_num}.csv")
        write_temp_csv(sample, temp_input)
        
        # Run classification
        temp_output = Path(f"temp_results_{batch_num}.csv")
        cmd = [
            sys.executable,
            "gemini_analysis.py",
            str(temp_input),
            "--output", str(temp_output),
            "--sample-fraction", "1.0",  # Classify all in this batch
            "--random-seed", str(seed)
        ]
        
        print(f"Running classification...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Classification failed: {result.stderr}")
            print("Stopping batch processing")
            break
        
        print(result.stdout)
        
        # Append results
        append_results(temp_output, args.output)
        
        # Clean up temp files
        if temp_input.exists():
            temp_input.unlink()
        if temp_output.exists():
            temp_output.unlink()
        
        # Update count
        classified_set = get_already_classified_paragraphs(args.output)
        new_total = len(classified_set)
        added = new_total - total_classified
        print(f"Added {added} classifications (total now: {new_total})")
        
        batch_num += 1
        
        if new_total >= args.target_count:
            print(f"\nReached target of {args.target_count} classifications!")
            break
        
        # Wait before next batch
        print(f"\nWaiting {args.wait_time} seconds before next batch...")
        time.sleep(args.wait_time)
    
    # Final summary
    final_count = len(get_already_classified_paragraphs(args.output))
    print(f"\n{'='*50}")
    print(f"Final classification count: {final_count}")
    print(f"Saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
