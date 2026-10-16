import time                                                            # NEW
import torch
from config import (block_size, batch_size, lr, max_iters,
                    eval_interval, eval_iters)                         # NEW: 2 more settings
from model import Model

torch.manual_seed(1337)                                                # NEW: same random numbers every run

if torch.cuda.is_available():
    device = "cuda"      # NVIDIA GPU (your pod)
elif torch.backends.mps.is_available():
    device = "mps"       # Apple GPU (your Mac)
else:
    device = "cpu"
print("device:", device, flush=True)                                   # NEW

text = open("./input.txt", encoding="utf-8").read()
vocab = sorted(set(text))
id_to_char = {}
char_to_id = {}

for idx, char in enumerate(vocab):
    id_to_char[idx]=char
    char_to_id[char]=idx


def encode(str):
    ret =[]
    for s in str:
        ret.append(char_to_id[s])
    return ret

def decode(list):
    s = ""
    for l in list:
        s+=id_to_char[l]
    return s

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

def get_batch(split):
    if split == "train":
        data = train_data
    else:
        data = test_data
    idx = torch.randint(len(data)-block_size-1, (batch_size,))

    l_x = []
    l_y = []
    for i in idx:
        l_x.append(data[i: i+block_size])
        l_y.append(data[i+1:i+block_size+1])

    x = torch.stack(l_x)
    y = torch.stack(l_y)

    x= x.to(device)
    y = y.to(device)
    return x, y

# MOVED: estimate_loss must be defined BEFORE the training loop uses it
@torch.no_grad()
def estimate_loss():
    train_loss = 0
    val_loss = 0
    model.eval()

    for _ in range(eval_iters):                                        # CHANGED: setting, not 200
        x, y = get_batch("train")
        x_v, y_v = get_batch("val")
        _, train_loss_t = model(x, y)
        _, val_loss_t = model(x_v, y_v)
        train_loss+=train_loss_t.item()
        val_loss+=val_loss_t.item()
    model.train()
    return train_loss/eval_iters, val_loss/eval_iters                  # CHANGED


model = Model()
model = model.to(device)                                               # NEW: without this, the model stays on the CPU
n_params = sum(p.numel() for p in model.parameters())                  # NEW
print(f"parameters: {n_params:,}", flush=True)                         # NEW

optimizer= torch.optim.AdamW(model.parameters(), lr)
t0 = time.time()                                                       # NEW
for i in range(max_iters):

    # NEW: measure, print and save every eval_interval steps, and at the last step
    if i % eval_interval == 0 or i == max_iters - 1:
        train_l, val_l = estimate_loss()
        mins = (time.time() - t0) / 60
        print(f"step {i:5d} | train {train_l:.4f} | val {val_l:.4f} | {mins:.1f} min", flush=True)
        torch.save(model.state_dict(), "ckpt.pt")

    x, y = get_batch("train")
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "ckpt.pt")                              # NEW: final weights
print("saved ckpt.pt", flush=True)                                     # NEW