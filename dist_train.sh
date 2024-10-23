CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=8888 \
    $(dirname "$0")/train_net.py \
    $CONFIG \
    --seed 0 \
    --deterministic \
    --launcher pytorch ${@:3}

