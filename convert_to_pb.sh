#usage : source convert_to_pb.sh

##################### Example of var_age_convert_to_lite.sh ######################
#INPUT_WIDTH=227
#INPUT_HEIGHT=227
#CKPT_PATH=/home/joyce-lin/myPython/rude-carnie/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_0/run-15
#CKPT=checkpoint-15000
#IS_QUANTIZE=false
#BAZEL_PATH=/home/joyce-lin/myPython/tensorflow
###############################################################################

source ../var_age_convert_to_lite.sh

echo "........... to freeze ..........."
$BAZEL_PATH/bazel-bin/tensorflow/python/tools/freeze_graph \
 --input_graph=$CKPT_PATH/model.pb \
 --input_checkpoint=$CKPT_PATH/$CKPT \
 --output_graph=$CKPT_PATH/graph_freeze.pb \
 --output_node_name="output/output"

echo "........... to opt(optimize_for_inference) ..........."
$BAZEL_PATH/bazel-bin/tensorflow/python/tools/optimize_for_inference \
  --input $CKPT_PATH/graph_freeze.pb \
  --output $CKPT_PATH/graph_optopt.pb \
  --input_names=tinydsod/inputs \
  --output_names=output/output
