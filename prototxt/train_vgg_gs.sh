
../caffe_kl/build/tools/caffe train \
    --solver=vgg_solver1.prototxt  --weights=../model/VGG_ILSVRC_16_layers.caffemodel -gpu 0 2>&1 | tee vgg_ldl_v1.txt
