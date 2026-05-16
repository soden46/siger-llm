# 🛠️ INSTALLATION.md — Panduan Instalasi MambaLM

## Requirement Sistem

| Komponen | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11+ |
| RAM | 4 GB | 8 GB |
| Storage | 5 GB | 20 GB |
| CPU | 2 core | 4 core |
| OS | Ubuntu 20.04 | Ubuntu 22.04 |
| GPU | Tidak wajib | NVIDIA RTX 3060+ |

---

## 1. Setup Environment

### Lokal / VPS Ubuntu

```bash
# Update sistem
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev
sudo apt install -y build-essential git curl

# Buat virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Windows (WSL2 recommended)

```bash
# Di PowerShell (jalanin sebagai admin):
wsl --install -d Ubuntu-22.04

# Lalu masuk ke WSL dan ikuti langkah Ubuntu di atas
```

---

## 2. Clone & Install Dependencies

```bash
# Clone repo
git clone https://github.com/yourname/mamba-llm.git
cd mamba-llm

# Install PyTorch CPU (untuk VPS tanpa GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install semua dependencies
pip install -r requirements.txt
```

### Kalau Ada GPU NVIDIA

```bash
# PyTorch dengan CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install bitsandbytes untuk INT4 quantization
pip install bitsandbytes

# Verifikasi CUDA
python -c "import torch; print(torch.cuda.is_available())"
# → True
```

---

## 3. Verifikasi Instalasi

```bash
# Test semua komponen
python -c "
import torch
import tiktoken
import onnxruntime as ort
import fastapi
import datasets

print('PyTorch     :', torch.__version__)
print('Tiktoken    : OK')
print('ORT         :', ort.__version__)
print('FastAPI     : OK')
print('Datasets    : OK')
print('Device      :', 'CUDA' if torch.cuda.is_available() else 'CPU')
"
```

Output yang diharapkan:
```
PyTorch     : 2.2.x+cpu
Tiktoken    : OK
ORT         : 1.17.x
FastAPI     : OK
Datasets    : OK
Device      : CPU
```

---

## 4. Test Model Pertama Kali

```bash
# Quick smoke test — pastiin model bisa forward pass
python -c "
import torch
from config.model_config import MambaConfig
from model.mamba_model   import MambaLM

config = MambaConfig(vocab_size=1000, d_model=128, n_layers=4)
model  = MambaLM(config)
x      = torch.randint(0, 1000, (2, 64))
logits, _ = model(x)
print('✅ Model OK | output shape:', logits.shape)
# → (2, 64, 1000)
"
```

```bash
# Test tokenizer
python -c "
from tokenizer.tokenizer import MultilingualTokenizer
tok  = MultilingualTokenizer()
ids  = tok.encode('Halo dunia!', add_bos=True, add_eos=True, lang='id')
text = tok.decode(ids)
print('✅ Tokenizer OK | tokens:', len(ids))
print('   Decoded:', text)
"
```

---

## 5. Setup VPS DewaCloud (Khusus)

### Spesifikasi Target
- **Provider:** Dewaweb DewaCloud
- **Spec:** 2 vCPU, 4 GB RAM, 50 GB SSD
- **OS:** Ubuntu 22.04 LTS

### Script Setup Otomatis

```bash
# Download dan jalanin setup script
curl -fsSL https://raw.githubusercontent.com/yourname/mamba-llm/main/setup_vps.sh | bash
```

Atau manual:

```bash
#!/bin/bash
# setup_vps.sh

set -e

echo "🚀 Setting up MambaLM on VPS..."

# System packages
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip \
                    git curl htop tmux build-essential

# Swap file 2GB (backup kalau RAM hampir habis)
if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "✅ Swap 2GB created"
fi

# Kernel optimizations untuk memory
sudo tee -a /etc/sysctl.conf << EOF
vm.swappiness=10
vm.vfs_cache_pressure=50
EOF
sudo sysctl -p

# Clone project
git clone https://github.com/yourname/mamba-llm.git
cd mamba-llm

# Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install CPU PyTorch (lebih kecil ~200MB vs GPU ~2GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "   Activate venv : source venv/bin/activate"
echo "   Start training: python main.py"
echo "   Start API     : python deploy.py"
```

### Setup Systemd Service (Auto-restart API)

```bash
# /etc/systemd/system/mamba-llm.service
sudo tee /etc/systemd/system/mamba-llm.service << EOF
[Unit]
Description=MambaLM API Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/mamba-llm
Environment="PATH=/home/$USER/mamba-llm/venv/bin"
ExecStart=/home/$USER/mamba-llm/venv/bin/python deploy.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mamba-llm
sudo systemctl start mamba-llm

# Cek status
sudo systemctl status mamba-llm
```

---

## 6. Setup Firewall (VPS)

```bash
# Allow SSH + API port
sudo ufw allow 22/tcp
sudo ufw allow 8000/tcp
sudo ufw enable

# Cek status
sudo ufw status
```

---

## 7. Struktur Direktori Setelah Instalasi

```
mamba-llm/
├── venv/                      # virtual environment (jangan di-commit)
├── checkpoints/               # model weights (jangan di-commit)
│   ├── tokenizer/
│   ├── lora/
│   └── onnx/
├── evaluation/
│   └── results/               # eval results JSON
└── ... (source code)
```

### .gitignore yang Direkomendasikan

```gitignore
venv/
__pycache__/
*.pyc
*.pyo
checkpoints/
*.pt
*.onnx
evaluation/results/
.env
*.log
```

---

## 8. Troubleshooting

### OOM (Out of Memory)

```bash
# Cek RAM usage
free -h
htop

# Solusi: kurangi batch size di config
# TRAIN_CONFIG["batch_size"] = 2  (dari 8)
# TRAIN_CONFIG["max_seq_len"] = 512  (dari 1024)
```

### ONNX Export Error

```bash
# Pastiin versi onnx compatible
pip install onnx==1.16.0 onnxruntime==1.17.0

# Test export ulang
python -c "import onnx; print(onnx.__version__)"
```

### Tiktoken Download Error

```bash
# Tiktoken butuh koneksi internet pertama kali untuk download encoding
# Kalau VPS tidak bisa akses internet, pre-download dulu:
python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
# Cache tersimpan di ~/.cache/huggingface/tiktoken/
```

### Port 8000 Sudah Dipakai

```bash
# Cari proses yang pakai port 8000
sudo lsof -i :8000
sudo kill -9 <PID>

# Atau ganti port di deploy.py:
# uvicorn.run(app, host="0.0.0.0", port=8080)
```