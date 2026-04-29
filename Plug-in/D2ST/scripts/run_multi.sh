#gpu=0
python runs/run.py \
    --cfg config/multisports/5-shot-train-0.3.yaml

python runs/run.py \
    --cfg config/multisports/5-shot-test-0.3.yaml
