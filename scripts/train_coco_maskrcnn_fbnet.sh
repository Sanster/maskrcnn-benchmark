export CUDA_VISIBLE_DEVICES=$1
echo 'Runing on GPU: '$1

python3 tools/train_net.py --config-file ./configs/e2e_mask_rcnn_fbnet.yaml