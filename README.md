#caffe学习笔记
安装,启动train过程,c++语言特性分析及学习

##caffe-gpu install
最终得到本地归档是caffe-gpu
env:Ubuntu16.04 caffe最新版 makefile
install:

安装依赖 hdf5 (apt 安装, 参考官网给出的依赖安装)
==修改cmake.config==
```sh
OPENCV_VERSION := 3
USE_CUDNN := 1
#CPU_ONLY := 1
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
DEBUG:=1
```
==修改 MakeFile==
```
CXXFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS) -std=c++11
NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS) -std=c++11
LINKFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS) -std=c++11
```
build:

sudo make clean

sudo make all -j2

sudo make test

sudo make runtest

sudo make pycaffe


##使用
```

dnn@DNN:~/caffe$ 
./examples/mnist/create_mnist.sh

dnn@DNN:~/caffe$ 
./examples/mnist/train_lenet.sh
```

##调试
```
gdb --args build/tools/caffe train --solver examples/mnist/lenet_solver.prototxt
```
CPU版本的这个调试不支持
性能测试:
```
./build/tools/==caffe time== --model='det/yolov3/yolov3.prototxt' --iterations=100 --gpu=0
```



##caffe train过程

tools/caffe.cpp 入口文件,参考该目录下的caffe.cpp,详细给出了注释和补充


##使用的c++特性
```
namespace
智能指针
工厂模式:Solver
内部类
友元
模板
&返回地址,没有&则返回值,&& 又值引用,一般的是左值引用
"##"和"#"
指针函数


```
