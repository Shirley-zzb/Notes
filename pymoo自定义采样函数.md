# pymoo自定义采样函数

在pymoo中，自定义随机取样的逻辑通常通过继承`Sampling`类并重写其`_do`方法来实现。以下是一个简单的示例，展示如何创建一个自定义的随机采样类：

```python
import numpy as np
from pymoo.model.sampling import Sampling

class MyCustomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        # 获取决策变量的维度和范围
        xl, xu = problem.bounds()
        
        # 生成随机样本
        X = np.random.uniform(xl, xu, (n_samples, len(xl)))
        
        return X

# 使用自定义采样方法
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("zdt1")

algorithm = MyCustomSampling()

res = minimize(problem,
               algorithm,
               ('n_gen', 40),
               seed=1,
               verbose=True)
```

在这个例子中，`MyCustomSampling`类继承了`Sampling`基类，并重写了`_do`方法来定义如何生成随机样本。在`_do`方法中，我们首先获取了问题的定义域（即决策变量的范围），然后使用numpy的`random.uniform`函数在这些范围内生成随机样本。最后，这些样本被返回用于进一步的处理或优化。
