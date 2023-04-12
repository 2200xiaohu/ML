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





# 快速获取字典value最大的key

```
max(my_dict, key=my_dict.get)
```



# 计算可哈希对象出现的次数

```
counts = collections.Counter(nums)
```

返回值：是一个字典，统计了每个对象出现的次数



# Sort（）

```
arr.sort() #从小到大
arr.sort(reverse=True)
```

**直接修改了原对象**



# Heap

**heapq 只支持创建最小堆，所以要创建最大堆可以用负数**

1、创建堆

```
heap = []
```

2、push

```
 heapq.heappush(heap , num)
```

3、pop

```
heapq.heappop(heap , num)
```

4、先push在pop

```
heapq.heappushpop(heap,num)
```

