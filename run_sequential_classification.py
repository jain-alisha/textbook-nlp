#!/usr/bin/env python3
"""
Run sequential classification in batches with wait times to manage API quota.
Processes paragraphs in order (not randomly) and appends results.
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def get_already_classified_count(output_csv: Path) -> int:
    """Count how many paragraphs have been classified."""
    if not output_csv.exists():
        return 0
    
    count = 0
    with output_csv.open('r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for _ in reader:
            count += 1
    return count


def get_total_paragraph_count(input_csv: Path) -> int:
    """Count total paragraphs in input CSV."""
    count = 0
    with input_csv.open('r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for _ in reader:
            count += 1
    return count


def create_sequential_batch(input_csv: Path, start_idx: int, batch_size: int, temp_output: Path):
    """Create a batch CSV with paragraphs starting from start_idx."""
    with input_csv.open('r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        all_paras = [row['paragraph'] for row in reader]
    
    end_idx = min(start_idx + batch_size, len(all_paras))
    batch = all_paras[start_idx:end_idx]
    
    with temp_output.open('w', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['paragraph'])
        for p in batch:
            writer.writerow([p])
    
    return len(batch)


def append_batch_results(temp_output: Path, final_output: Path):
    """Append new results to final output."""
    if not temp_output.exists():
        return 0
    
    with temp_output.open('r', encoding='utf8') as f:
        lines = f.readlines()
    
    # Skip header
    if lines and lines[0].startswith('paragraph,category'):
        lines = lines[1:]
    
    mode = 'a' if final_output.exists() else 'w'
    with final_output.open(mode, encoding='utf8') as f:
        if mode == 'w':
            f.write('paragraph,category\n')
        f.writelines(lines)
    
    return len(lines)


def main():
    parser = argparse.ArgumentParser(description="Sequential batch classification with wait times")
    parser.add_argument("input_csv", type=Path, help="Input paragraphs CSV")
    parser.add_argument("--output", type=Path, default=Path("analysis_results.csv"), help="Output CSV")
    parser.add_argument("--target-fraction", type=float, default=0.3, help="Fraction of total to classify (0-1)")
    parser.add_argument("--batch-size", type=int, default=10, help="Paragraphs per batch")
    parser.add_argument("--wait-time", type=int, default=60, help="Seconds to wait between batches")
    parser.add_argument("--clear", action="store_true", help="Clear existing results before starting")
    args = parser.parse_args()

    # Clear existing results if requested
    if args.clear and args.output.exists():
        args.output.unlink()
        print(f"Cleared existing results from {args.output}")
    
    # Calculate target
    total_paragraphs = get_total_paragraph_count(args.input_csv)
    target_count = int(total_paragraphs * args.target_fraction)
    
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Target: {target_count} ({args.target_fraction*100:.1f}%)")
    print(f"Batch size: {args.batch_size}")
    print(f"Wait time: {args.wait_time}s between batches")
    print()
    
    batch_num = 1
    current_count = get_already_classified_count(args.output)
    
    while current_count < target_count:
        print(f"=== Batch {batch_num} ===")
        print(f"Progress: {current_count}/{target_count} ({current_count/target_count*100:.1f}%)")
        
        # Create batch starting from current_count
        temp_input = Path(f"temp_batch_{batch_num}.csv")
        batch_size = min(args.batch_size, target_count - current_count)
        actual_batch_size = create_sequential_batch(args.input_csv, current_count, batch_size, temp_input)
        
        if actual_batch_size == 0:
            print("No more paragraphs to process")
            break
        
        print(f"Processing paragraphs {current_count} to {current_count + actual_batch_size}")
        
        # Run classification
        temp_output = Path(f"temp_results_{batch_num}.csv")
        cmd = [
            sys.executable,
            "gemini_analysis.py",
            str(temp_input),
            "--output", str(temp_output),
            "--sample-fraction", "1.0",
            "--random-seed", "42"
        ]
        
        print(f"Classifying...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Show output
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"  {line}")
        
        if result.returncode != 0 or "quota exhausted" in result.stdout.lower():
            print("\nAPI quota hit or error occurred")
            if "quota exhausted" in result.stdout.lower():
                print("Appending partial results and stopping...")
                append_batch_results(temp_output, args.output)
            print("You can resume later by running the same command (without --clear)")
            break
        
        # Append results
        added = append_batch_results(temp_output, args.output)
        current_count = get_already_classified_count(args.output)
        
        print(f"Added {added} classifications")
        print(f"Total now: {current_count}/{target_count}")
        
        # Clean up
        if temp_input.exists():
            temp_input.unlink()
        if temp_output.exists():
            temp_output.unlink()
        
        batch_num += 1
        
        if current_count >= target_count:
            print(f"\n✓ Reached target of {target_count} classifications!")
            break
        
        # Wait before next batch
        print(f"\nWaiting {args.wait_time} seconds before next batch...\n")
        time.sleep(args.wait_time)
    
    # Final summary
    final_count = get_already_classified_count(args.output)
    print(f"\n{'='*60}")
    print(f"Classification complete!")
    print(f"Total classified: {final_count}/{total_paragraphs} ({final_count/total_paragraphs*100:.1f}%)")
    print(f"Saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
