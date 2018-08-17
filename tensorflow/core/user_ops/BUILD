load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "tc_op.so",
    srcs = ["tc_op.cc"],
    deps = ["@tensor_comprehensions//:tc_core"],
)
