#gpu=0
python runs/run.py \
    --cfg config/multisports/5-shot-train.yaml

python runs/run.py \
    --cfg config/multisports/5-shot-test.yaml
