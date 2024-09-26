# 多个数据集，对应多个机器学习模型，统一进行训练、对比
from abc import abstractmethod, ABCMeta

# 抽象产品：定义机器学习模型的接口
class mlmodel(metaclass=ABCMeta):
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass

# 具体产品：实现具体的机器学习模型
class modela(mlmodel):
    def train(self):
        print("training model a")
    
    def predict(self):
        print("predicting with model a")
    
    def evaluate(self):
        print("evaluating model a")

class modelb(mlmodel):
    def train(self):
        print("training model b")
    
    def predict(self):
        print("predicting with model b")
    
    def evaluate(self):
        print("evaluating model b")

# 抽象创建者：定义抽象工厂类
class modelfactory(metaclass=ABCMeta):
    @abstractmethod
    def get_model(self):
        pass

# 具体创建者：实现具体工厂类
class factorya(modelfactory):
    def get_model(self):
        return modela()

class factoryb(modelfactory):
    def get_model(self):
        return modelb()

# 客户端使用
def clientcode(factory):
    model = factory.get_model()
    model.train()
    model.predict()
    model.evaluate()

# 运行客户端代码
clientcode(factorya())
clientcode(factoryb())
