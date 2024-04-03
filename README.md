# Mistral with EETQ float16->8bit quantization

> used: https://github.com/NetEase-FuXi/EETQ

## Installation

```
# 1. install mistral dependencies
pip install -r requirements.txt

# 2. install eetq
git clone https://github.com/NetEase-FuXi/EETQ.git
cd EETQ/
git submodule update --init --recursive
pip install .
```

## Download the model
```
wget https://models.mistralcdn.com/mistral-7b-v0-1/mistral-7B-v0.1.tar (md5sum: 37dab53973db2d56b2da0a033a15307f)
tar -xf mistral-7B-v0.1.tar
```

## Run the model

```
python -m main demo /path/to/mistral-7B-v0.1/
```
## Result on 3090

- TTFT: Time To First Token
- TPOT: Time Per Output Token 

|           | Before     | After               |
|-----------|------------|---------------------|
| TTFT      | 0.30s      | 0.31s               |
| TPOT      | 0.026s     | 0.016s              |
|           | [BFLOAT16] | [FLOAT16 + 8bit]    |



