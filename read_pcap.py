import pyshark

PCAP_FILE = './20210115-000005-00459768.mia.pcap'
READ_LINES = 100

if __name__ == '__main__':
print
    i = 0
    for pkt in cap:
        i+=1
        if i > READ_LINES:
            break
        if 'IP' in pkt:
            print('source:',pkt.ip.src,'destination:', pkt.ip.dst)
            
