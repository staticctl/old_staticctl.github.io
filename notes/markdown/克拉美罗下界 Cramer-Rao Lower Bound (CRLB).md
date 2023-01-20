## 克拉美罗下界 Cramer-Rao Lower Bound (CRLB)

链接：[计算例子](https://blog.csdn.net/u013701860/article/details/78154069?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-78154069-blog-88715517.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-78154069-blog-88715517.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=1)、[简要介绍](https://blog.csdn.net/GongPF/article/details/88715517?spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-88715517-blog-78154069.pc_relevant_3mothn_strategy_and_data_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-88715517-blog-78154069.pc_relevant_3mothn_strategy_and_data_recovery&utm_relevant_index=9)

论文：[Analyzing complex single-molecule emission patterns with deep learning](https://pubmed.ncbi.nlm.nih.gov/30377349/)

克拉美罗下界$ Cramer-Rao Lower Bound (CRLB)$可以用于计算无偏估计中能够获得的最佳估计精度，因此经常用于计算理论能达到的最佳估计精度，和评估参数估计方法的性能（是否接近$CRLB$下界）

计算$CRLB$的方法：

1. 构建$N$的观测量$x[n]$与估计参数θ的联合概率密度函数 $p(X ; θ)=Π p(x[n] ; θ)$，然后求对数，得到对数似然函数$\ln p(\mathbf{x} ; \theta)$

2. 用对数似然函数对参数$θ$求二阶导数$\partial^2 \ln p(\mathbf{x} ; \theta) / \partial \theta^2$

3. 如果结果依赖于$x[n]$，则求期望，否则跳过。这个期望就是**费雪信息**。

4. $$
   -\left.E\left\{\frac{\partial^2 \ln p(\mathbf{x} ; \theta)}{\partial \theta^2}\right\}\right|_{\theta=\text { true value }}
   $$

5. 连续函数则为：

6. $$
   -E\left\{\frac{\partial^2 \ln p(\mathbf{x} ; \theta)}{\partial \theta^2}\right\}=-\int \frac{\partial^2 \ln p(\mathbf{x} ; \theta)}{\partial \theta^2} p(x ; \theta) d x
   $$

7. **注意这里的期望是仅仅是对每个$x[n]$求取的，同时该期望不是求这N个观测量的平均，而是理论期望。**例如，如果$x[n]$为依赖于$n$的正太分布$N(k_n , σ^2)$，那么$x[n]$的期望就是$kn$。如果不知道期望，应该也可以就用$x[n]$近似，因为其期望也是其若干采样的平均。

8. 求费雪信息的倒数即可得到$CRLB$下界，$\operatorname{var}\{\hat{\theta}\}=\frac{1}{I(\theta)}=C R L B$

9. 实际的$CRLB$可能跟参数$θ$本身有关，那么则需要先知道（预估）$θ$（仿真的时候一般知道实际参数值，实际实验应该只能先估计$θ$），再计算其预估精度$CRLB$。

   

