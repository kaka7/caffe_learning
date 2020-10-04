源码阅读 
solver/layer 工厂类 
参考 https://zhuanlan.zhihu.com/p/25102232和~/git/marathon/pattern
Blob：是基础的数据结构，是用来保存学习到的参数以及网络传输过程中产生数据的类。
Layer：是网络的基本单元，由此派生出了各种层类。修改这部分的人主要是研究特征表达方向的。Caffe支持CUDA，在数据级别上也做了一些优化，这部分最重要的是知道它主要是对protocol buffer所定义的数据结构的继承，Caffe也因此可以在尽可能小的内存占用下获得很高的效率。
Net：是网络的搭建，将Layer所派生出层类组合成网络。
Solver：是Net的求解，修改这部分人主要会是研究DL求解方向的。

作者：北川谦一
链接：https://www.zhihu.com/question/27982282/answer/39350629
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

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



![效果](./result.JPG)

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
