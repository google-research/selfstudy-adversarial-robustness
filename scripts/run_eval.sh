echo "========================================================="
echo "Running attack defense_baseline/attack_linf.py"
python3 evaluate.py defense_baseline/attack_linf.py --test
echo "========================================================="
echo "Running attack defense_quantize/attack_linf.py"
python3 evaluate.py defense_quantize/attack_linf.py --test
echo "========================================================="
echo "Running attack defense_temperature/attack_linf.py"
python3 evaluate.py defense_temperature/attack_linf.py --test
echo "========================================================="
echo "Running attack defense_mergebinary/attack_linf.py"
python3 evaluate.py defense_mergebinary/attack_linf.py --test
echo "========================================================="
echo "Running attack defense005_autoencoder/attack_linf.py"
python3 evaluate.py defense005_autoencoder/attack_linf.py --test
echo "========================================================="
echo "Running attack defense_thermometer/attack_linf.py"
python3 evaluate.py defense_thermometer/attack_linf.py --test
echo "========================================================="
echo "Running attack defense_injection/attack_linf.py"
python3 evaluate.py defense_injection/attack_linf.py --test
echo "========================================================="
echo "Running attack defense_randomneuron/attack_linf.py"
python3 evaluate.py defense_randomneuron/attack_linf.py --test
echo "========================================================="
echo "Running attack defense_jump/attack_linf.py"
python3 evaluate.py defense_jump/attack_linf.py --test

echo "========================================================="
echo "Running attack defense_baseline/attack_linf_torch"
python3 evaluate.py defense_baseline/attack_linf_torch.py --test
echo "========================================================="
echo "Running attack defense_quantize/attack_linf_torch"
python3 evaluate.py defense_quantize/attack_linf_torch.py --test
echo "========================================================="
echo "Running attack defense_temperature/attack_linf_torch"
python3 evaluate.py defense_temperature/attack_linf_torch.py --test
echo "========================================================="
echo "Running attack defense_mergebinary/attack_linf_torch"
python3 evaluate.py defense_mergebinary/attack_linf_torch.py --test
echo "========================================================="
echo "Running attack defense005_autoencoder/attack_linf_torch"
python3 evaluate.py defense005_autoencoder/attack_linf_torch.py --test
echo "========================================================="
echo "Running attack defense_thermometer/attack_linf_torch"
python3 evaluate.py defense_thermometer/attack_linf_torch.py --test
echo "========================================================="
echo "Running attack defense_injection/attack_linf_torch"
python3 evaluate.py defense_injection/attack_linf_torch.py --test
echo "========================================================="
echo "Running attack defense_randomneuron/attack_linf_torch"
python3 evaluate.py defense_randomneuron/attack_linf_torch.py --test
echo "========================================================="
echo "Running attack defense_jump/attack_linf_torch"
python3 evaluate.py defense_jump/attack_linf_torch.py --test
