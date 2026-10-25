import torch, time
from transformers import MambaForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained('state-spaces/mamba-130m-hf')
ids = tok("The quick brown fox", return_tensors="pt").input_ids.cuda()

def step_time(name):
    m = MambaForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    pos = torch.tensor([ids.shape[1]]).cuda()
    with torch.no_grad():
        out = m(ids, use_cache=True)
        cache, nxt = out.cache_params, ids[:, -1:]
        for _ in range(5):
            m(nxt, cache_params=cache, use_cache=True, cache_position=pos)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(30):
            m(nxt, cache_params=cache, use_cache=True, cache_position=pos)
        torch.cuda.synchronize()
    dt = (time.time() - t) / 30
    print(name, f"{dt*1000:.1f} ms/token ({1/dt:.1f} tok/s)")
    del m
    torch.cuda.empty_cache()
    return dt

d = step_time('state-spaces/mamba-130m-hf')
t = step_time('state-spaces/mamba-1.4b-hf')
print(f"draft/target cost ratio c = {d/t:.2f}")