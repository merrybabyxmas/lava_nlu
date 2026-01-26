# LAVA


Before installing LAVA, you need to set up the PEFT_LAVA dependency:

```bash
git clone -b no_gate https://github.com/merrybabyxmas/peft_lava.git
cd peft_lava
chmod +x setup.sh
./setup.sh
cd ..

git clone -b no_logit https://github.com/merrybabyxmas/lava.git
cd lava

conda activate lava
pip install -r requirements.txt

# 4. for rtx 5090 / pro6000 ...etc might need upgraded version of torch
pip uninstall torch torchvision torchaudio -y

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall --no-cache-dir
```
