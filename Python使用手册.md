# np.random.choice

根据给出的概率分布“P”选择索引

```
np.random.seed(0) 
p = np.array([0.1, 0.0, 0.7, 0.2]) 
index = np.random.choice([0, 1, 2, 3], p = p.ravel())
```



# watch -n 1 nvidia-smi

每隔1秒查看nvidia-smi



# OS

## os.walk(path)

遍历path下的所有目录

```
返回值
(root, dirs, files)
root：当前目录
dirs：是一个list，当前目录下所有目录的名字（不包括子目录）
files：当前目录下的所有文件

例如：
Data
|
|--train
	|
	|---0
	|	|- a.png
	|	|-b.png
	|	
	|---1
	
os.walk("Data\\train")
返回值依次为；
(Data\\train, 0, files)
(Data\\train, 1, files)
(Data\\train\\0, null, files)
(Data\\train\\1, null, files)
```



## os.listdir(path)

返回path下所有文件的路径
