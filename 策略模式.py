# 处理不同条件下的数据，采用不同策略
from abc import ABC, abstractmethod

# 定义策略接口
class DataPreprocessingStrategy(ABC):
    @abstractmethod
    def preprocess(self, data):
        pass

# 实现具体策略
class PreprocessingStrategyA(DataPreprocessingStrategy):
    def preprocess(self, data):
        # 实现针对数据集A的预处理逻辑
        print(f'用策略A处理数据{data}')
        pass

class PreprocessingStrategyB(DataPreprocessingStrategy):
    def preprocess(self, data):
        # 实现针对数据集B的预处理逻辑
        print(f'用策略B处理数据{data}')
        pass

# 创建上下文context类，用来切换、调用类
class DataPreprocessor:
    def __init__(self, strategy: DataPreprocessingStrategy):
        self._strategy = strategy

    def preprocess_data(self, data):
        return self._strategy.preprocess(data)

# 客户端代码
# 假设有一个数据集dataset_a和对应的预处理策略PreprocessingStrategyA
dataset_a = 'dataset_a'
preprocessor = DataPreprocessor(PreprocessingStrategyA())
preprocessed_data_a = preprocessor.preprocess_data(dataset_a)

# 假设有一个数据集dataset_b和对应的预处理策略PreprocessingStrategyB
dataset_b = 'dataset_b'
preprocessor = DataPreprocessor(PreprocessingStrategyB())
preprocessed_data_b = preprocessor.preprocess_data(dataset_b)
