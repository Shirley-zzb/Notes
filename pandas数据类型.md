# pandas数据类型

## 1、DataFrame

二维数据，整个表格，多行多列

pandas直接读取excel返回就是DataFrame类型

- 表格型数据结构，每列可以是不同类型

- 既有行索引index，也有列索引columns
- 可以看成series组成的字典

```python
data = {
    'index1':[1,2,3],
    'index2':[6,7,8]
}
#key值默认为列索引，行索引默认为0开始的序列
data.columns #返回列索引
data.index #返回行索引
```

## 2、Series

一维数据，一行或一列，类似一维数组

```python
s = pd.Series([1,'a',5.2,7],index = ['a','b','c','d'])
#两列，第一列是index，第二列是数据
s.index #获取索引
s.values #获取数据
```

可以用字典创建

```python
s = {'a':1,'b':2}
s = pd.Series(s)
```

查询方法

```python
s['a'] #查询一个值，返回原生值
s[['a','b']] #查询多个，返回仍然是一个series
```

## 3、关系

```python
data['index1'] #查询结果为一个series
data[['index1','index2']] #查询多列是dataframe
data.loc[1] #查询一行，返回一个series
data.loc[0:1] #查询多行，返回一个dataframe，包含末尾元素
```





















