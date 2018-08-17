First time creation of environment

```
# prefer the same version of python as system python3, otherwise there may be
# some quirks in the build
conda create -y --name tctf python=3.5
conda activate tctf
conda install -y pyyaml mkl-include pytest
conda install -y -c nicolasvasilache llvm-trunk halide
conda install -y -c pytorch pytorch=0.4.0 torchvision cuda90
conda remove -y cudatoolkit --force
conda remove -y protobuf libprotobuf
pip install --upgrade pip

# get Keras applications (see https://github.com/tensorflow/tensorflow/issues/21518)
pip install keras_applications==1.0.4 --no-deps
pip install keras_preprocessing==1.0.2 --no-deps
```

Actual build
```
conda activate tctf

# make sure to select NO for XLA as it brings LLVM dependence in the following command
# and point it to conda python binary and package path
./configure

# build TC opeator with TF sources
bazel build --config opt --config monolithic //tensorflow/core/user_ops:tc_op.so

# clean previous builds
rm -rf /tmp/tensorflow_pkg

# build TF python package without XLA because it depends on a different
# configuration of LLVM than TC
bazel build --config=opt -- //tensorflow/tools/pip_package:build_pip_package -//tensorflow/compiler/...

# install python package
pip install /tmp/tensorflow_pkg/tensorflow-*.whl

# follow https://www.tensorflow.org/extend/adding_an_op#use_the_op_in_python
# to load the op in python
```

