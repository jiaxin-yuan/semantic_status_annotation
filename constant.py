from pathlib import Path


PATTERNS = ["hard", "llm"]
DATAS = ["bpic2017a", "bpic2017c", "bpic2017r", "production", "traffic", "bpic20d", "bpic20i", "bpic20permit", "bpic20prepaid", "bpic20request", "helpdesk", "bpic19"]

PATHS_M_SUFFIX = ["maps/bpic17", "maps/bpic17", "maps/bpic17", "maps/production", "maps/traffic", "maps/bpic20d", "maps/bpic20i", "maps/bpic20permit", "maps/bpic20prepaid", "maps/bpic20request", "maps/helpdesk", "maps/bpic19"]
PATHS_D = ["datas/BPIC17_O_Accepted.csv", "datas/BPIC17_O_Cancelled.csv", "datas/BPIC17_O_Refused.csv", "datas/Production.csv", "datas/traffic_fines_1.csv", "datas/bpic20d.csv", "datas/bpic20i.csv", "datas/bpic20permit.csv", "datas/bpic20prepaid.csv", "datas/bpic20request.csv", "datas/helpdesk.csv", "datas/bpic19.csv"]
PATHS_D_H = ["datas/BPIC17_O_Accepted_hard.csv", "datas/BPIC17_O_Cancelled_hard.csv", "datas/BPIC17_O_Refused_hard.csv", "datas/product_hard.csv", "datas/traffic_hard.csv", "datas/bpic20d_hard.csv", "datas/bpic20i_hard.csv", "datas/bpic20permit_hard.csv", "datas/bpic20prepaid_hard.csv", "datas/bpic20request_hard.csv", "datas/helpdesk_hard.csv", "datas/bpic19_hard.csv"]
PATHS_D_L = ["datas/BPIC17_O_Accepted_llama.csv", "datas/BPIC17_O_Cancelled_llama.csv", "datas/BPIC17_O_Refused_llama.csv", "datas/product_llama.csv", "datas/traffic_llama.csv", "datas/bpic20d_llama.csv", "datas/bpic20i_llama.csv", "datas/bpic20permit_llama.csv", "datas/bpic20prepaid_llama.csv", "datas/bpic20request_llama.csv", "datas/helpdesk_llama.csv", "datas/bpic19_llama.csv"]

MAP_PATHS = {data: path for data, path in zip(DATAS, PATHS_M_SUFFIX)}
DATA_PATHS = {data: path for data, path in zip(DATAS, PATHS_D)}
DATA_PATHS_H = {data: path for data, path in zip(DATAS, PATHS_D_H)}
DATA_PATHS_L = {data: path for data, path in zip(DATAS, PATHS_D_L)}