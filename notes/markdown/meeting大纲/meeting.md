meeting

存在的问题

![image-20221212104258856](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20221212104258856.png)

![](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20221211224102640.png)

1. 为什么不使用**matlab**自带的互相关？可能是精度问题？（猜测）
2. **FFT2**不需要并行，无论是**GPU**还是**CPU**，可以分理运行，忽略不计的时间
3. 是否提前开辟空间也很重要，out = function 或者 先定义sim，sim = function，这两种时间差距也很大
4. 如何拍摄到**x-z**方向上的**PSF**

![image-20221220000624027](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20221220000624027.png)