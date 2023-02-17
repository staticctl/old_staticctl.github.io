# Fast, background-free, 3D super-resolution optical fluctuation imaging (SOFI)



## Summary

写完笔记之后最后填，概述文章的内容，以后查阅笔记的时候先看这一段。注：写文章summary切记需要通过自己的思考，用自己的语言描述。忌讳直接Ctrl + c原文。



## Research Objective(s)

提升时间序列的分辨率,使用累加和的形式



## Background / Problem Statement

研究的背景以及问题陈述：作者需要解决的问题是什么？



## Method(s)

作者解决问题的方法/算法是什么？是否基于前人的方法？基于了哪些？



## Evaluation

作者如何评估自己的方法？实验的setup是什么样的？感兴趣实验数据和结果有哪些？有没有问题或者可以借鉴的地方？



## Conclusion

作者给出了哪些结论？哪些是strong conclusions, 哪些又是weak的conclusions（即作者并没有通过实验提供evidence，只在[discussion](https://www.zhihu.com/search?q=discussion&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A142802496})中提到；或实验的数据并没有给出充分的evidence）?



## Notes

#### 理论推导

给定由$N$个独立发射器组成的样本，每个发射器的位置$r_{K}$，随时间变化的分子亮度为


$$
\varepsilon_{k}·s_{k}
$$
荧光源的分布为
$$
\sum_{k=1}^N \delta\left(\mathbf{r}-\mathbf{r}_k\right) \cdot \varepsilon_k s_k(t)
$$
$\varepsilon_{k}$是恒定的分子亮度，$s_k(t)$是随时间变化的波动

假设辐射源在采集过程中不变化

为了简单起见，我们进一步假设点扩散函数$( PSF )$不因像差或极化效应而在局部变化。

在位置$\mathbf{r}$和时间$t$的荧光信号$F(\mathbf{r}, t)$，由系统的$PSF$     $U(\mathbf{r})$的卷积和荧光源分布的卷积给出

也就是在位置$\mathbf{r}$和时间$t$的荧光信号$F(\mathbf{r}, t)$，是由系统PSF 和 荧光源分布的卷积得出的，下文公式中的**·**是卷积的意思
$$
F(\mathbf{r}, t)=\sum_{k=1}^N U\left(\mathbf{r}-\mathbf{r}_k\right) \cdot \varepsilon_k s_k(t)
$$
假设样本在采集过程中处于平稳均衡状态，波动可以表示为零均值波动：
$$
\begin{aligned}
\delta F(\mathbf{r}, t) & =F(\mathbf{r}, t)-\langle F(\mathbf{r}, t)\rangle_t \\
& =\sum_k U\left(\mathbf{r}-\mathbf{r}_k\right) \cdot \varepsilon_k \cdot\left[s_k(t)-\left\langle s_k(t)\right\rangle_t\right] \\
& =\sum_k U\left(\mathbf{r}-\mathbf{r}_k\right) \cdot \varepsilon_k \cdot \delta s_k(t),
\end{aligned}
$$
$\langle\ldots\rangle_t$为时间平均

补充知识：[归一化、标准化、零均值化作用及区别](https://zhuanlan.zhihu.com/p/183591302)

二阶自相关函数：
$$
\begin{aligned}
\mathrm{G}_2(\mathbf{r}, \tau) & =\langle\delta \mathrm{F}(\mathbf{r}, t+\tau) \cdot \delta \mathrm{F}(\mathbf{r}, t)\rangle_t \\
& =\sum_{j, k} U\left(\mathbf{r}-\mathbf{r}_j\right) U\left(\mathbf{r}-\mathbf{r}_k\right) \cdot \varepsilon_j \cdot \varepsilon_k \cdot\left\langle\delta s_l(t+\tau) \delta s_k(t)\right\rangle \\
& =\sum_k U^2\left(\mathbf{r}-\mathbf{r}_k\right) \cdot \varepsilon_k^2 \cdot\left\langle\delta s_k(t+\tau) s_k(t)\right\rangle
\end{aligned}
$$

$$
\begin{aligned}
& U(\mathbf{r})=\exp \left(-\frac{x^2+y^2}{2 \omega_0^2}-\frac{z^2}{2 \omega_{z 0}^2}\right) \\
\Rightarrow & U^2(\mathbf{r})=\exp \left(-\frac{x^2+y^2}{2 \tilde{\omega}_0^2}-\frac{z^2}{2 \tilde{\omega}_{0 z}^2}\right)
\end{aligned}
$$

with $\tilde{\omega}_{0 z}=\omega_{0 z} / \sqrt{2}$ and $\tilde{\omega}_0=\omega_0 / \sqrt{2}$















## References



