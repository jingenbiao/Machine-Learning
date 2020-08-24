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
## 决策树
### 决策树的介绍
决策树是一种常见的分类模型，在金融风控、医疗辅助诊断等诸多行业具有较为广泛的应用。决策树的核心思想是基于树结构对数据进行划分，这种思想是人类处理问题时的本能方法。例如在婚恋市场中，女方通常会先询问男方是否有房产，如果有房产再了解是否有车产，如果有车产再看是否有稳定工作……最后得出是否要深入了解的判断。</br>
决策树的主要优点：</br>
1.具有很好的解释性，模型可以生成可以理解的规则。</br>
2.可以发现特征的重要程度。</br>
3.模型的计算复杂度较低。</br>

决策树的主要缺点：</br>
1.模型容易过拟合，需要采用减枝技术处理。</br>
2.不能很好利用连续型特征。</br>
3.预测能力有限，无法达到其他强监督模型效果。</br>
4.方差较高，数据分布的轻微改变很容易造成树结构完全不同。</br>

### 决策树的应用
由于决策树模型中自变量与因变量的非线性关系以及决策树简单的计算方法，使得它成为集成学习中最为广泛使用的基模型。梯度提升树(GBDT)，XGBoost以及LightGBM等先进的集成模型都采用了决策树作为基模型，在广告计算、CTR预估、金融风控等领域大放异彩，成为当今与神经网络相提并论的复杂模型，更是数据挖掘比赛中的常客。在新的研究中，南京大学周志华教授提出一种多粒度级联森林模型，创造了一种全新的基于决策树的深度集成方法，为我们提供了决策树发展的另一种可能。

同时决策树在一些明确需要可解释性或者提取分类规则的场景中被广泛应用，而其他机器学习模型在这一点很难做到。例如在医疗辅助系统中，为了方便专业人员发现错误，常常将决策树算法用于辅助病症检测。例如在一个预测哮喘患者的模型中，医生发现测试的许多高级模型的效果非常差。在他们运行了一个决策树模型后发现，算法认为剧烈咳嗽的病人患哮喘的风险很小。但医生非常清楚剧烈咳嗽一般都会被立刻检查治疗，这意味着患有剧烈咳嗽的哮喘病人都会马上得到收治。用于建模的数据认为这类病人风险很小，是因为所有这类病人都得到了及时治疗，所以极少有人在此之后患病或死亡。

### 算法实战
#### 算法构建的伪代码
输入： 训练集D={( 𝑥1 , 𝑦1 ),( 𝑥2 , 𝑦2 ),....,( 𝑥𝑚 , 𝑦𝑚 )};
特征集A={ 𝑎1 , 𝑎2 ,...., 𝑎𝑑 }

输出： 以node为根节点的一颗决策树

过程：函数TreeGenerate( 𝐷 , 𝐴 )

1.生成节点node</br>
2.𝑖𝑓   𝐷 中样本全书属于同一类别 𝐶   𝑡ℎ𝑒𝑛 :</br>
3.----将node标记为 𝐶 类叶节点； 𝑟𝑒𝑡𝑢𝑟𝑛 </br>
4.𝑖𝑓   𝐴  = 空集 OR D中样本在 𝐴 上的取值相同  𝑡ℎ𝑒𝑛 </br>:
5.----将node标记为叶节点，其类别标记为 𝐷 中样本数最多的类； 𝑟𝑒𝑡𝑢𝑟𝑛 </br>
6.从  𝐴  中选择最优划分属性  𝑎∗ ;</br>
7.𝑓𝑜𝑟   𝑎∗  的每一个值  𝑎𝑣∗   𝑑𝑜 :</br>
8.----为node生成一个分支，令 𝐷𝑣 表示 𝐷 中在 𝑎∗ 上取值为 𝑎𝑣∗ 的样本子集；</br>
9.---- 𝑖𝑓   𝐷𝑣  为空  𝑡ℎ𝑒𝑛 :</br>
10.--------将分支节点标记为叶节点，其类别标记为 𝐷 中样本最多的类; 𝑡ℎ𝑒𝑛 </br>
11.---- 𝑒𝑙𝑠𝑒 :</br>
12.--------以 TreeGenerate( 𝐷𝑣 , 𝐴 { 𝑎∗ })为分支节点</br>
决策树的构建过程是一个递归过程。函数存在三种返回状态：</br>
（1）当前节点包含的样本全部属于同一类别，无需继续划分；</br>
（2）当前属性集为空或者所有样本在某个属性上的取值相同，无法继续划分；</br>
（3）当前节点包含的样本集合为空，无法划分。</br>
#### 划分选择
从上述伪代码中我们发现，决策树的关键在于line6.从 𝐴 中选择最优划分属性 𝑎∗ ，一般我们希望决策树每次划分节点中包含的样本尽量属于同一类别，也就是节点的“纯度”更高
##### 信息增益
信息熵是一种衡量数据混乱程度的指标，信息熵越小，则数据的“纯度”越高

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>Ent</mi>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <mo stretchy="false">(</mo>
  <mi>D</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mo>&#x2212;<!-- − --></mo>
  <munderover>
    <mo>&#x2211;<!-- ∑ --></mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>k</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mrow class="MJX-TeXAtom-ORD">
        <mo stretchy="false">|</mo>
      </mrow>
      <mrow class="MJX-TeXAtom-ORD">
        <mi class="MJX-tex-caligraphic" mathvariant="script">Y</mi>
      </mrow>
      <mrow class="MJX-TeXAtom-ORD">
        <mo stretchy="false">|</mo>
      </mrow>
    </mrow>
  </munderover>
  <msub>
    <mi>p</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>k</mi>
    </mrow>
  </msub>
  <msub>
    <mi>log</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mn>2</mn>
    </mrow>
  </msub>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <msub>
    <mi>p</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>k</mi>
    </mrow>
  </msub>
</math>
</br>其中 𝑝𝑘 代表了第 𝑘 类样本在 𝐷 中占有的比例。

假设离散属性 𝑎 有 𝑉 个可能的取值{ 𝑎1 , 𝑎2 ,...., 𝑎𝑉 }，若使用 𝑎 对数据集 𝐷 进行划分，则产生 𝐷 个分支节点，记为 𝐷𝑣 。则使用 𝑎 对数据集进行划分所带来的信息增益被定义为：

Gain(𝐷,𝑎)=Ent(𝐷)−∑𝑉𝑣=1||𝐷𝑣|||𝐷|Ent(𝐷𝑣) 
</br>一般的信息增益越大，则意味着使用特征 𝑎 来进行划分的效果越好。
##### 基尼系数
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>Gain</mi>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <mo stretchy="false">(</mo>
  <mi>D</mi>
  <mo>,</mo>
  <mi>a</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mi>Ent</mi>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <mo stretchy="false">(</mo>
  <mi>D</mi>
  <mo stretchy="false">)</mo>
  <mo>&#x2212;<!-- − --></mo>
  <munderover>
    <mo>&#x2211;<!-- ∑ --></mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>v</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>V</mi>
    </mrow>
  </munderover>
  <mfrac>
    <mrow>
      <mo>|</mo>
      <msup>
        <mi>D</mi>
        <mrow class="MJX-TeXAtom-ORD">
          <mi>v</mi>
        </mrow>
      </msup>
      <mo>|</mo>
    </mrow>
    <mrow>
      <mrow class="MJX-TeXAtom-ORD">
        <mo stretchy="false">|</mo>
      </mrow>
      <mi>D</mi>
      <mrow class="MJX-TeXAtom-ORD">
        <mo stretchy="false">|</mo>
      </mrow>
    </mrow>
  </mfrac>
  <mi>Ent</mi>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <mrow>
    <mo>(</mo>
    <msup>
      <mi>D</mi>
      <mrow class="MJX-TeXAtom-ORD">
        <mi>v</mi>
      </mrow>
    </msup>
    <mo>)</mo>
  </mrow>
</math></br>
基尼指数反映了从数据集 𝐷 中随机抽取两个的类别标记不一致的概率。</br>
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>Gini</mi>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <mi>index</mi>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <mo stretchy="false">(</mo>
  <mi>D</mi>
  <mo>,</mo>
  <mi>a</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <munderover>
    <mo>&#x2211;<!-- ∑ --></mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>v</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>V</mi>
    </mrow>
  </munderover>
  <mfrac>
    <mrow>
      <mo>|</mo>
      <msup>
        <mi>D</mi>
        <mrow class="MJX-TeXAtom-ORD">
          <mi>v</mi>
        </mrow>
      </msup>
      <mo>|</mo>
    </mrow>
    <mrow>
      <mrow class="MJX-TeXAtom-ORD">
        <mo stretchy="false">|</mo>
      </mrow>
      <mi>D</mi>
      <mrow class="MJX-TeXAtom-ORD">
        <mo stretchy="false">|</mo>
      </mrow>
    </mrow>
  </mfrac>
  <mi>Gini</mi>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <mrow>
    <mo>(</mo>
    <msup>
      <mi>D</mi>
      <mrow class="MJX-TeXAtom-ORD">
        <mi>v</mi>
      </mrow>
    </msup>
    <mo>)</mo>
  </mrow>
</math></br>
使用特征 𝑎 对数据集 𝐷 划分的基尼指数定义为上。
##### 重要参数
###### criterion
Criterion这个参数正是用来决定模型特征选择的计算方法的。sklearn提供了两种选择：

输入”entropy“，使用信息熵（Entropy）

输入”gini“，使用基尼系数（Gini Impurity）
###### random_state & splitter
random_state用来设置分枝中的随机模式的参数，默认None，在高维度时随机性会表现更明显。splitter也是用来控制决策树中的随机选项的，有两种输入值，输入”best"，决策树在分枝时虽然随机，但是还是会优先选择更重要的特征进行分枝（重要性可以通过属性feature_importances_查看），输入“random"，决策树在分枝时会更加随机，树会因为含有更多的不必要信息而更深更大，并因这些不必要信息而降低对训练集的拟合。
###### max_depth
限制树的最大深度，超过设定深度的树枝全部剪掉。这是用得最广泛的剪枝参数，在高维度低样本量时非常有效。决策树多生长一层，对样本量的需求会增加一倍，所以限制树深度能够有效地限制过拟合。
###### min_samples_leaf
min_samples_leaf 限定，一个节点在分枝后的每个子节点都必须包含至少min_samples_leaf个训练样本，否则分枝就不会发生，或者，分枝会朝着满足每个子节点都包含min_samples_leaf个样本的方向去发生。一般搭配max_depth使用，在回归树中有神奇的效果，可以让模型变得更加平滑。这个参数的数量设置得太小会引起过拟合，设置得太大就会阻止模型学习数据。

## 支持向量机
### 介绍
支持向量机（Support Vector Machine，SVM）是一个非常优雅的算法，具有非常完善的数学理论，常用于数据分类，也可以用于数据的回归预测中，由于其其优美的理论保证和利用核函数对于线性不可分问题的处理技巧，在上世纪90年代左右，SVM曾红极一时。

我们常常会碰到这样的一个问题，首先给你一些分属于两个类别的数据，现在需要一个线性分类器，将这些数据分开来。我们可能会有多种分法：那么现在有一个问题，两个分类器，哪一个更好呢？为了判断好坏，我们需要引入一个准则：好的分类器不仅仅是能够很好的分开已有的数据集，还能对未知数据集进行两个的划分。</br>

那么如何客观的评判两条线的健壮性呢？此时，我们需要引入一个非常重要的概念：最大间隔。最大间隔刻画着当前分类器与数据集的边界，那么，我们现在的分类器是最优分类器吗？或者说，有没有更好的分类器，它具有更大的间隔？答案是有的。为了找出最优分类器，我们需要引入我们今天的主角：SVM</br>

带黑边的点是距离当前分类器最近的点，我们称之为支持向量。支持向量机为我们提供了在众多可能的分类器之间进行选择的原则，从而确保对未知数据集具有更高的泛化性。</br>

### 软间隔
很多情况并不容易找到这样的最大间隔。于是我们就有了软间隔，相比于硬间隔而言，我们允许个别数据出现在间隔带中。我们知道，如果没有一个原则进行约束，满足软间隔的分类器也会出现很多条。所以需要对分错的数据进行惩罚，SVC 函数中，有一个参数 C 就是惩罚参数。惩罚参数越小，容忍性就越大。

### 超平面
我们可以将二维（低维）空间的数据映射到三维（高维）空间中。此时，我们便可以通过一个超平面对数据进行划分所以，我们映射的目的在于使用 SVM 在高维空间找到超平面的能力。</br>
在 SVC 中，我们可以用高斯核函数来实现这以功能：kernel='rbf'.此时便完成了非线性分类。
