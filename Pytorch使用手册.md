# argparse.ArgumentParser

==Sample==

```
parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=15, help='steps to run')
opt = parser.parse_args()
```



**使用流程：**

1、创建解析器

```
parser=[argparse](https://so.csdn.net/so/search?q=argparse&spm=1001.2101.3001.7020).ArgumentParser()
```



2、添加参数

```
parser.add_argument()
```

每个参数解释：

name or flags:字符串的名字或者列表。

action:当参数在命令行中出现时使用的动作。

nargs：应该读取的命令行参数个数

const：不指定参数时的默认值

type：命令行参数应该被转换成的类型

choices：参数可允许的值的另一个容器

required：可选参数是否可省略

help：参数的帮助信息

metavar - 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.

dest - 解析后的参数名称，默认情况下，对于可选参数选取最长的名称，中划线转换为下划线


3、解析参数

```
parser.parse_args()
```



4、调用参数

```
exp_path = paras.exppath
```



# Optimize

当需要定制不同层的不同学习率的时候，需要满足

1、传进去参数组不要有空

