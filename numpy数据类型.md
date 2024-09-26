# numpy

## numpy数据类型

约等于数组类型

```python
a = np.array([1,2,3])
print(a, type(a))

>>> [1 2 3] <class 'numpy.ndarray'>
```

默认情况是 ndarray 类型

```python
numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)

#object:数组或嵌套的数列
#dtype:固定元素的数据类型 b:布尔型 i:整型 f:浮点型
#copy:是否深度复制，默认支持深度复制
#subick:创建的时候是否保持原来数据类型
#ndmin:指定创建最小维度
```







