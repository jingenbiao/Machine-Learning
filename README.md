# Machine-Learning
========================================
## 逻辑回归原理
当z≥0 时,y≥0.5,分类为1，当 z<0时,y<0.5,分类为0，其对应的y值我们可以视为类别1的概率预测值。Logistic回归虽然名字里带“回归”，但是它实际上是一种分类方法，主要用于两分类问题（即输出只有两种，分别代表两个类别），所以利用了Logistic函数（或称为Sigmoid函数），函数形式为：<br>
![](https://img.alicdn.com/tfs/TB1VkG.eP39YK4jSZPcXXXrUFXa-180-88.jpg)<br>
对应的函数图像可以表示如下:<br>
```python
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-5,5,0.01)
y = 1/(1+np.exp(-x))

plt.plot(x,y)
plt.xlabel('z')
plt.ylabel('y')
plt.grid()
plt.show()
```
![](https://img.alicdn.com/tfs/TB1Ou1JNXY7gK0jSZKzXXaikpXa-386-262.png)<br>
通过上图我们可以发现 Logistic 函数是单调递增函数，并且在z=0，将回归方程写入其中为：<br>
![](https://img.alicdn.com/tfs/TB1_t3vNkY2gK0jSZFgXXc5OFXa-503-89.jpg)<br>
所以，<br>
![](https://img.alicdn.com/tfs/TB1yNckNhD1gK0jSZFsXXbldVXa-517-46.jpg)<br>
逻辑回归从其原理上来说，逻辑回归其实是实现了一个决策边界：对于函数![](https://img.alicdn.com/tfs/TB1B0kjNbY1gK0jSZTEXXXDQVXa-94-51.jpg)，当z≥0 时,y≥0.5,分类为1，当 z<0时,y<0.5,分类为0，其对应的y值我们可以视为类别1的概率预测值。对于模型的训练而言：实质上来说就是利用数据求解出对应的模型的特定的ω。从而得到一个针对于当前数据的特征逻辑回归模型。而对于多分类而言，将多个二分类的逻辑回归组合，即可实现多分类。
