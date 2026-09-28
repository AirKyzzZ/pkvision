import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")
app = modal.App("pkvision-test", image=image)

@app.function(gpu="T4", timeout=120)
def gpu_check():
    import torch
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"GPU: {name}, VRAM: {mem:.1f} GB, CUDA: {torch.version.cuda}"

@app.local_entrypoint()
def main():
    result = gpu_check.remote()
    print(result)
