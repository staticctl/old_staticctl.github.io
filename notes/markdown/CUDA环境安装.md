## CUDA环境安装

**Abstract:** 主要为各种用户提供CUDA编程环境
**Keywords:** 腾讯云，CUDA，GPU云，windows，vs

> Windows CUDA环境安装

现在的 **Linux** 环境 对于**CUDA** 开发更友好，但是还是有很多场景需要**Windows**开发环境，在**Windows**下首先考虑的开发环境就是**Visual Studio**，本文以 **Visual Studio** 为例，描述 **Windows** 下的 **CUDA** 开发环境安装。

首先确保你的电脑有 **Nvdia** 显卡，在安装驱动后

1. 安装**Visual Studio 2019**，本人开发 **CUDA** 用于 **Matlab mex** 函数，**Matlab** 支持 **Visual Studio 2019**
2. 安装 **CUDA**，安装默认设置安装即可
3. 注意安装的顺序，必须先安装 **VS** 再 安装 **CUDA**

> Linux CUDA 环境安装

有时候很多大型计算的服务器使用的 **Linux** 系统，所以要在 **Linux** 上安装 **CUDA** 环境，一般个人 **PC** 不配置 **Linux** 系统 ，我们可以考虑配置云服务器，购买过程省略，链接服务器过程省略

1. 一般会默认安装驱动，输入`NVCC`测试
2. 安装 **CMAKE**
3. 下载 **CMAKE**源码并解压，进入源目录
4. 执行`./bootstrap`
5. 执行`make`
6. 执行`sudo make install`

> 验证环境 

执行 `nvdia-smi`，**Linux**和**Windows** 都适用，验证是否正确安装驱动

测试一个小例子，看是否安装成功

从大神的代码库中**clone**一份我们的学习**cuda**的代码，**cmake**一下

`git clone https://github.com/Tony-Tan/CUDA_Freshman.git`