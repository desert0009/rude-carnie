#usage : source benchmark.sh

#======== Example of var_age_benchmark.sh ========
#GRAPH_PATH=/home/joyce-lin/myPython/rude-carnie/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_0/run-9002/graph_opt.pb
#INPUT_WIDTH=227
#INPUT_HEIGHT=227
#BAZEL_PATH=/home/joyce-lin/myPython/tensorflow
#============================================

source ../var_age_benchmark.sh

echo "........... summarize_graph ..........."
echo "[GRAPH]: $GRAPH_PATH"
$BAZEL_PATH/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
  --in_graph=$GRAPH_PATH

echo "........... benchmark_model ..........."
$BAZEL_PATH/bazel-bin/tensorflow/tools/benchmark/benchmark_model \
  --graph=$GRAPH_PATH \
  --input_layer="tinydsod/inputs" \
  --input_layer_shape="1, $INPUT_WIDTH, $INPUT_HEIGHT, 3" \
  --input_layer_type="float" \
  --output_layer="output/output" \
  --show_run_order=false \
  --show_time=false \
  --show_memory=false \
  --show_summary=true \
  --show_flops=true

