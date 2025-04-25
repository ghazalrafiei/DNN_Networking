import numpy as np
import pandas as pd
from scapy.all import RawPcapReader, Ether, IP, IPv6, UDP, DNS, DNSQR

import csv
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse

# data preprocess

def preprocess_pcap(pcap_file: str, csv_file: str):
    """
    Extract DNS query packets from a pcap file and save to CSV.
    Fields: domain, timestamp, qtype, rcode, src_ip
    Only includes DNS query (qr=0) packets.
    """

    count = 0
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'timestamp', 'qtype', 'rcode', 'src_ip'])

        reader = RawPcapReader(pcap_file)
        for pkt_data, pkt_metadata in reader:
            try:
                eth = Ether(pkt_data)
                if eth.haslayer(IP):
                    pkt = eth[IP]
                elif eth.haslayer(IPv6):
                    pkt = eth[IPv6]
                else:
                    continue

                if not (pkt.haslayer(UDP) and pkt.haslayer(DNS)):
                    continue

                dns = pkt[DNS]
                if dns.qr != 0 or dns.qdcount == 0:
                    continue

                domain = dns.qd.qname.decode(errors='ignore').rstrip('.').lower()
                if not domain or '.' not in domain:
                    continue

                timestamp = datetime.fromtimestamp(pkt_metadata.sec + pkt_metadata.usec / 1e6)
                qtype = dns.qd.qtype
                rcode = dns.rcode
                src_ip = pkt.src

                writer.writerow([domain, timestamp, qtype, rcode, src_ip])
                count += 1
            except Exception:
                continue

    print(f"[Done] Extracted {count} DNS query records to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", required=True, help="Path to pcap file")
    parser.add_argument("--csv", required=True, help="Output CSV file path")
    args = parser.parse_args()

    preprocess_pcap(args.pcap, args.csv)


