# Getting started

## Installation

1. Optionally create Python virtual environment.

```bash
# Create virtual environtment
virtualenv -p python3 --system-site-packages ~/.venv3/advx_selfstudy
# Activate virtual environtment
source ~/.venv3/advx_selfstudy/bin/activate
```
NOTE: to deactivate virtualenv after you done use `deactivate` command.

2. Clone repository with the self-study.

```bash
git clone https://github.com/google-research/selfstudy-adversarial-robustness.git
cd selfstudy-adversarial-robustness
```

3. Add the root directory of repository to the python path, e.g., with

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

4. Install the dependencies.

```bash
pip install -r requirements.txt
```

5. Optionally install PyTorch.

```bash
pip install torch~=1.7
```

6. Prepare checkpoints for all models used in the self-study. You can either download them or train them yourself.

To download all pre-trained checkpoints (TensorFlow+PyTorch) use the following command:

```bash
wget https://github.com/google-research/selfstudy-adversarial-robustness/releases/download/v0.1/checkpoints.tgz
tar -xzf checkpoints.tgz
```

Just the TensorFlow or PyTorch models can be downloaded from the paths `tf_checkpoints.tgz` or `torch_checkpoints.tgz` for a decreased download size.


To train TensorFlow versions of all models from scratch use the following command:

```bash
# Without arguments train_all.sh will run training in parallel on all available GPUs
training/train_all.sh

# If you want to use only specific GPUs, then provide their IDs using --gpus argument.
# You can list all your GPUs with their IDs using nvidia-smi command.
training/train_all.sh --gpus=0,1
```

If you're using PyTorch you can convert your TensorFlow checkpoints into PyTorch with the following command:

```bash
python3 convert_pytorch.py checkpoints/
```

## Computing benign model test accuracy

Start by evaluating a baseline model's accuracy

```bash
python evaluate.py --test defense_baseline/attack_linf.py
```

You should see an output something like

```
Evaluation parameters:
  Defense path:  /path/to/source/defense_baseline
  Attack name:  attack_linf.py
  Dataset:  cifar10
  Number of examples: 100
  Defense test accuracy 0.92
```

You can run the same code using a pytorch-backed model instead of TensorFlow by running 

```bash
python evaluate.py --test defense_baseline/attack_linf_torch.py
```

## Running a first attack

Attacks are run with the same script, but removing the --test flag.
For example, modify the `defense_baseline/attack_linf.py` script as follows
to add noise each input example

```diff
diff --git a/public/defense_baseline/attack_linf.py b/public/defense_baseline/attack_linf.py
index cf829e7..f7ff96a 100644
--- a/public/defense_baseline/attack_linf.py
+++ b/public/defense_baseline/attack_linf.py
@@ -22,8 +22,8 @@ class LinfAttack(common.framework.Attack):
     def attack(self, model, x, y):
         # TODO: Write your attack code here
         # You can query model by calling `model(x)`
-
-        return x
+        import numpy as np
+        return x + np.sign(np.random.normal(size=x.shape))*self.task.threshold

```

And then evaluate by running

```bash
python evaluate.py defense_baseline/attack_linf.py
```

The attack probably does not succeed (much) by just adding random noise. Can you do better?
