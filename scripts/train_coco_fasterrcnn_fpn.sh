


export CUDA_VISIBLE_DEVICES=$1
echo 'Runing on GPU: '$1
 
python3 tools/train_net.py --config-file configs_my/e2e_faster_rcnn_R_50_FPN_1x.yaml 
