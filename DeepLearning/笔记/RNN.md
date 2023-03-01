# RNN简单结构

1、每一层都会考虑前一层的情况，相当于每一层都会有两个输入，一个是前一层的a，一个是新输入的词

2、计算每一层的 a 有三个参数$W_{aa},W_{ax},b_{a}$ 
$$
a^{<t>}=g(W_{aa}a^{t-1}+W_{ax}x^{<t>}+b_{a})
$$
![image-20230301111252615](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230301111252615.png)