# 关于torch中构建的神经网络模型的二次深刻理解——从设计模式的角度

从设计模式出发，设计多个神经网络模型集成环境下的模型时，对于神经网络模型的构建比较困难。不像sklearn中已经封装好的机器学习模型，集成度很高，树模型涉及到的超参数也比较统一，深度、数的数量等。而神经网络模型，不同模型之间的区别较大，往往由多个小的、不同的神经网络单元构成一个复杂的网络模型，因此涉及到的超参数往往种类不同，数量较多。

第二点，从构建模型的类上看，实现神经网络模型要比实现sklearn中已经封装好的机器学习模型要复杂，常见的主要模式是，现在init函数中对继承模块的小结构单元、网络层等进行继承/获取，然后在forward中进行计算逻辑的实现。当采用设计模式时，又往往需要一个单独的方法来创建神经网络模型，gpt当时给出的代码是直接将这个函数用于了神经网络模型的实现过程，即定义init和forward的类。

在我之前的抽象工厂设计模式中，创建模型的方法是对模型类实例化，从而返回模型，这样不会造成这样的问题。也就是说，我需要单独定义多个神经网络模型的方法，然后进行实例化就可以。

如果没有这样做，采用gpt的方式，就会造成一定错误，具体如下：

```
可是之前是采用了工厂模式，如果把forward更改后就会报错，我是不是能将creat_model中返回forward函数的闭包？
Traceback (most recent call last):

  File D:\anaconda\lib\site-packages\spyder_kernels\py3compat.py:356 in compat_exec
    exec(code, globals, locals)

  File d:\desktop\研究生工作\发表论文\参与论文      未完成\27 宗壮师兄论文\论文240915\code\model\dl.py:182
    model = clientcode(PytorchModelFactory, d, f'reactor {conditions[i]}.pkl', searchDict)

  File d:\desktop\研究生工作\发表论文\参与论文      未完成\27 宗壮师兄论文\论文240915\code\model\dl.py:157 in clientcode
    model = factory.create_model('lstm', 2, 50, 2) # input_size = 10, hidden_size = 50, num_layers = 2

  File d:\desktop\研究生工作\发表论文\参与论文      未完成\27 宗壮师兄论文\论文240915\code\model\dl.py:144 in create_model
    return LSTMModel(*args)  # 传入input_size, hidden_size, num_layers

TypeError: Can't instantiate abstract class LSTMModel with abstract method create_model
```

我认识到，gpt采用的创建模型为什么能取代forward函数我一直不理解，现在才看明白。神经网络模型进行计算的时候其实就是自动调用forward这样一个计算函数，而这一过程如果直接放在create_model中就无法自动调用了。

于是gpt采用了`nn.Sequential` 的使用，来将整个计算过程也就是forward进行了包装，这或许对简单的dnn或者是cnn是没问题的，但是这里用的是LSTM的复杂模型，这也让我意识到，很多复杂模型无法通过简单的nn.Sequential来解决。我想到，既然我需要的是返回一个函数，那么是不是就可以用闭包来返回模型的forward逻辑，于是尝试了gpt也给出的两种解决方法。但是很明显，仅仅将forward函数赋值给self.model将整个模型的逻辑可能都破坏了，说明模型也不仅仅是简单的返回forward函数，而是在不断的调用forward。

这样做模型的参数不会注册到网络中，从而导致优化过程找不到参数。



参考：本文中我们通过一些实例学习了 ModuleList 和 Sequential 这两种 nn containers，ModuleList  就是一个储存各种模块的 list，这些模块之间没有联系，没有实现 forward 功能，但相比于普通的 Python  list，ModuleList 可以把添加到其中的模块和参数自动注册到网络上。而Sequential  内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部 forward 功能已经实现，可以使代码更加整洁。

https://zhuanlan.zhihu.com/p/64990232

https://zhuanlan.zhihu.com/p/75206669