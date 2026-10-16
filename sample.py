
import sys
import torch
from model import Model

torch.manual_seed(42)        # change this number to get a different sample

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ---- the same vocabulary as in train.py (it must match exactly) ----
text = open("./input.txt", encoding="utf-8").read()
vocab = sorted(set(text))
char_to_id = {c: i for i, c in enumerate(vocab)}
id_to_char = {i: c for i, c in enumerate(vocab)}

def encode(s):
    return [char_to_id[c] for c in s]

def decode(ids):
    return "".join(id_to_char[i] for i in ids)

# ---- settings from the command line ----
n_new = int(sys.argv[1]) if len(sys.argv) > 1 else 500
start = sys.argv[2] if len(sys.argv) > 2 else "\n"

# ---- load the model ----
model = Model()
state = torch.load("ckpt.pt", map_location=device)   # map_location: works on any machine
model.load_state_dict(state)
model = model.to(device)
model.eval()                                          # turns dropout off

# ---- generate ----
idx = torch.tensor([encode(start)], dtype=torch.long, device=device)   # shape (1, T)
with torch.no_grad():
    out = model.generate(idx, n_new)

print(decode(out[0].tolist()))