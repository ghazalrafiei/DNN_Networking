import os
import re
import gzip
import glob
import logging
import subprocess
import shutil
import concurrent.futures
from datetime import datetime
from collections import defaultdict
import pandas as pd
import argparse

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "/nas/eclairnas01/users/shefalit/working/data"
OUTPUT_DIR = "csv_files"
TEMP_DIR = "temp_csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

FILE_PATTERN = re.compile(r'(\d{8})-(\d{6})-\d+\.lax\.pcap\.xz')

def second_data_default():
    return {
        'packet_count': 0,
        'total_size': 0,
        'flows': set(),
        'tcp_count': 0,
        'udp_count': 0
    }

def parse_second(ts):
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y%m%d-%H%M%S")
    except Exception as e:
        logger.warning(f"Bad timestamp {ts}: {e}")
        return None

def process_file_to_temp_csv(file):
    basename = os.path.basename(file)
    logger.info(f"STARTED processing: {basename}")
    match = FILE_PATTERN.match(basename)
    if not match:
        logger.warning(f"Skipping unrecognized filename: {basename}")
        return None

    day = match.group(1)[-2:]
    temp_output = os.path.join(TEMP_DIR, f"{basename}.csv")
    cmd = f"xzcat '{file}' | tcpdump -tt -nn -r -"
    second_data = defaultdict(second_data_default)

    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            executable="/bin/bash"
        )

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                ts_float = float(parts[0])
                ts = parse_second(ts_float)
                if not ts:
                    continue
            except Exception:
                logger.warning(f"Bad timestamp line: {line}")
                continue

            size = 64
            src = parts[2]
            dst = parts[4].rstrip(':')
            proto = parts[1]

            stats = second_data[ts]
            stats['packet_count'] += 1
            stats['total_size'] += size
            stats['flows'].add(f"{src}->{dst}")
            if proto == 'IP':
                stats['tcp_count'] += 1
            elif proto == 'UDP':
                stats['udp_count'] += 1

        rows = []
        for second, stats in second_data.items():
            pkt_count = stats['packet_count']
            avg_size = stats['total_size'] / pkt_count if pkt_count else 0
            rows.append({
                'day': day,
                'timestamp': second,
                'packet_count': pkt_count,
                'avg_packet_size': round(avg_size, 2),
                'flow_count': len(stats['flows']),
                'tcp_count': stats['tcp_count'],
                'udp_count': stats['udp_count']
            })

        pd.DataFrame(rows).to_csv(temp_output, index=False)
        logger.info(f"✅ Wrote temp CSV: {temp_output}")
        return (day, temp_output)

    except Exception as e:
        logger.error(f"Error processing {file}: {str(e)}")
        return None

def merge_temp_csvs(day, temp_files):
    dfs = []
    for f in temp_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")
    if not dfs:
        logger.warning(f"No data to merge for day {day}")
        return

    full_df = pd.concat(dfs)
    grouped = full_df.groupby('timestamp').agg({
        'packet_count': 'sum',
        'avg_packet_size': 'mean',
        'flow_count': 'sum',
        'tcp_count': 'sum',
        'udp_count': 'sum'
    }).reset_index()

    output_path = os.path.join(OUTPUT_DIR, f"{day}_preprocessed_data.csv.gz")
    with gzip.open(output_path, 'wt') as f:
        grouped.to_csv(f, index=False)
    logger.info(f"📦 Merged CSV for day {day}: {output_path}")

def clean_temp_dir():
    try:
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        logger.info(f"🧹 Cleaned up temp directory: {TEMP_DIR}")
    except Exception as e:
        logger.warning(f"Failed to clean temp directory: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=str, required=True, help="Specify day to process (e.g., 08, 09, 10)")
    parser.add_argument('--test', action='store_true', help="Run on 1 file only")
    args = parser.parse_args()

    target_day = args.day
    matched_files = [
        f for f in glob.glob(os.path.join(DATA_DIR, "*.lax.pcap.xz"))
        if f"202101{target_day}" in os.path.basename(f)
    ]

    if not matched_files:
        logger.error(f"No files found for day {target_day}")
        return

    logger.info(f"Found {len(matched_files)} files for day {target_day}")

    if args.test:
        logger.info("TEST MODE: Processing 1 file")
        result = process_file_to_temp_csv(matched_files[0])
        if result:
            day, temp_file = result
            merge_temp_csvs(day, [temp_file])
            clean_temp_dir()
        return

    temp_csvs = []
    logger.info("🚀 Starting parallel processing with 4 workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(process_file_to_temp_csv, f): f for f in matched_files}
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                if result:
                    _, temp_file = result
                    temp_csvs.append(temp_file)
            except Exception as exc:
                logger.error(f"❌ Exception for file {file}: {exc}")

    merge_temp_csvs(target_day, temp_csvs)
    clean_temp_dir()

if __name__ == "__main__":
    main()

