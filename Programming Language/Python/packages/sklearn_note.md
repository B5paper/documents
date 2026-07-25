# scikit-learn Note

* url cache:

    * sklearn pca

        <https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py>

    * sklearn example

        <https://scikit-learn.org/stable/auto_examples/index.html#general-examples>

    * sklearn random forest

        <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>

## cache

* 详细讲讲 sklearn 中的 KFold

	好的，我们来详细讲讲 `sklearn` 中的 `KFold`。这是机器学习中最基础也最重要的交叉验证工具之一。

	---

	## 1. 什么是 KFold？

	`KFold` 是 **K 折交叉验证**（K-Fold Cross-Validation）的实现。  
	它的核心思想是：

	> 将整个数据集平均分成 **K 份（折）**，然后依次选择其中 **1 份作为验证集**，其余 **K-1 份作为训练集**，重复 K 次，最终得到 K 个模型评估结果。

	---

	## 2. 为什么需要 KFold？

	- **避免固定划分带来的偏差**（比如只分一次训练/测试集可能不够稳定）
	- **充分利用有限数据**：每个样本都会被用作训练和验证
	- **更稳定地评估模型泛化能力**（取 K 次结果的平均值）

	---

	## 3. 基本用法

	### 导入
	```python
	from sklearn.model_selection import KFold
	```

	### 创建对象
	```python
	kf = KFold(n_splits=5, shuffle=False, random_state=None)
	```

	**参数说明**：
	- `n_splits`：折数 K（通常取 5 或 10）
	- `shuffle`：是否在划分前打乱数据（推荐 True，防止顺序影响）
	- `random_state`：随机种子（shuffle=True 时生效，保证可复现）

	---

	## 4. 核心方法

	| 方法 | 作用 |
	|------|------|
	| `split(X, y=None)` | 生成训练/验证索引的迭代器 |
	| `get_n_splits()` | 返回折数 |

	---

	## 5. 代码示例

	```python
	import numpy as np
	from sklearn.model_selection import KFold

	X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
	y = np.array([0, 1, 0, 1, 0, 1])

	kf = KFold(n_splits=3, shuffle=True, random_state=42)

	for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
		print(f"Fold {fold+1}")
		print("Train indices:", train_idx)
		print("Val indices:", val_idx)
		print("X_train:\n", X[train_idx])
		print("X_val:\n", X[val_idx])
		print("------")
	```

	**输出示例**（因为 shuffle=True，顺序随机）：
	```
	Fold 1
	Train indices: [0 1 2 4]
	Val indices: [3 5]
	...
	```

	---

	## 6. 在模型训练中实际使用

	通常配合交叉验证分数使用，也可以手动训练：

	```python
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score

	kf = KFold(n_splits=5, shuffle=True, random_state=0)
	scores = []

	for train_idx, val_idx in kf.split(X, y):
		X_train, X_val = X[train_idx], X[val_idx]
		y_train, y_val = y[train_idx], y[val_idx]
		
		model = LogisticRegression()
		model.fit(X_train, y_train)
		pred = model.predict(X_val)
		scores.append(accuracy_score(y_val, pred))

	print("平均准确率:", np.mean(scores))
	```

	但更常见的是直接用 `cross_val_score`：

	```python
	from sklearn.model_selection import cross_val_score

	scores = cross_val_score(LogisticRegression(), X, y, cv=KFold(5, shuffle=True))
	print(scores.mean())
	```

	---

	## 7. KFold 与 StratifiedKFold 的区别

	| | KFold | StratifiedKFold |
	|------|-------|-----------------|
	| 划分方式 | 随机平均分成 K 份 | 保持每一折中各类别比例与整体一致 |
	| 适用场景 | 回归、类别均衡的分类 | **分类任务（尤其类别不均衡）** |
	| 推荐度 | 通用 | **分类问题优先推荐** |

	示例：
	```python
	from sklearn.model_selection import StratifiedKFold
	skf = StratifiedKFold(n_splits=5, shuffle=True)
	```

	---

	## 8. 重要注意事项

	1. **数据顺序敏感**  
	   如果数据按标签排序（如先全 0 后全 1），且 `shuffle=False`，可能导致某一折全是一个类别 → 模型失效。  
	   **建议始终 `shuffle=True`**。

	2. **时间序列数据不能用 KFold**  
	   因为未来信息不能用来预测过去 → 应使用 `TimeSeriesSplit`。

	3. **KFold 只返回索引，不复制数据**  
	   内存友好。

	4. **K 值选择**  
	   - K 太小（如 2）→ 验证集太大，评估方差大  
	   - K 太大（如 50）→ 训练集太相似，评估偏差大  
	   - 常用：5 或 10

	---

	## 9. 进阶：KFold 与 GroupKFold / LeaveOneOut

	| 变体 | 用途 |
	|------|------|
	| `GroupKFold` | 保证同一组数据不会同时出现在训练和验证集（如病人数据） |
	| `LeaveOneOut` | K = 样本数，每次留一个样本（适合小数据） |
	| `RepeatedKFold` | 重复多次 KFold（增加稳定性） |

	---

	## 10. 总结一句话

	> **KFold 是 sklearn 中最标准的交叉验证划分器，通过将数据均匀分成 K 份并轮流作为验证集，来稳定评估模型性能；实际使用时建议 shuffle=True，分类任务优先用 StratifiedKFold。**

	---

	如果你还想了解：
	- 如何与 Pipeline 结合  
	- 如何在 KFold 中自定义数据预处理（避免数据泄露）  
	- 或 KFold 与 GridSearchCV 的配合细节  

	我可以继续展开讲，你告诉我方向即可。

* TimeSeriesSplit

	好的，我们来详细讲解 `sklearn` 中的 `TimeSeriesSplit`。这是专门为**时间序列数据**设计的交叉验证工具，和 `KFold` 有本质区别。

	---

	## 1. 为什么需要 TimeSeriesSplit？

	**核心问题**：时间序列数据具有**时间顺序依赖性**（如股票价格、气温、销量）。

	- **KFold 的问题**：随机打乱数据会破坏时间顺序，导致**用未来数据预测过去**（数据泄露），评估结果虚假乐观。
	- **TimeSeriesSplit 的解决方案**：**始终保证训练集在验证集之前**，模拟真实预测场景。

	---

	## 2. 基本原理

	`TimeSeriesSplit` 采用**扩展窗口**（Expanding Window）策略：

	- 第 1 折：训练集 = `[0 : t1]`，验证集 = `[t1 : t2]`
	- 第 2 折：训练集 = `[0 : t2]`（扩展，包含之前所有数据），验证集 = `[t2 : t3]`
	- 第 3 折：训练集 = `[0 : t3]`，验证集 = `[t3 : t4]`
	- ...依此类推

	**示意图**：
	```
	折1: [训练][验证]
	折2: [训练  ][验证]
	折3: [训练    ][验证]
	折4: [训练      ][验证]
	```

	**注意**：训练集不断**累积扩展**，而不是滑动窗口（滑动窗口是另一种方法，sklearn 未直接提供）。

	---

	## 3. 基本用法

	### 导入
	```python
	from sklearn.model_selection import TimeSeriesSplit
	```

	### 创建对象
	```python
	tscv = TimeSeriesSplit(
		n_splits=5,           # 折数
		max_train_size=None,  # 每折训练集最大样本数（None表示不限制）
		test_size=None        # 每折验证集大小（None表示自动计算）
	)
	```

	**参数详解**：
	- `n_splits`：要生成的折数（注意：实际折数可能等于 n_splits，但取决于数据量）
	- `max_train_size`：限制训练集最大尺寸（用于早期数据过多时，防止训练过慢）
	- `test_size`：固定每折验证集大小（默认 None 时自动分配剩余数据）

	---

	## 4. 核心方法与属性

	| 方法/属性 | 作用 |
	|-----------|------|
	| `split(X, y=None, groups=None)` | 生成训练/验证索引迭代器 |
	| `get_n_splits(X, y=None, groups=None)` | 返回实际折数 |

	---

	## 5. 代码示例

	### 基础示例
	```python
	import numpy as np
	from sklearn.model_selection import TimeSeriesSplit

	X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
	y = np.array([1, 2, 3, 4, 5, 6])

	tscv = TimeSeriesSplit(n_splits=3)

	for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
		print(f"Fold {fold+1}")
		print(f"训练索引: {train_idx} -> {X[train_idx]}")
		print(f"验证索引: {val_idx} -> {X[val_idx]}")
		print("---")
	```

	**输出**：
	```
	Fold 1
	训练索引: [0 1] -> [[1 2] [3 4]]
	验证索引: [2 3] -> [[5 6] [7 8]]
	---
	Fold 2
	训练索引: [0 1 2 3] -> [[1 2] [3 4] [5 6] [7 8]]
	验证索引: [4 5] -> [[9 10] [11 12]]
	---
	Fold 3
	训练索引: [0 1 2 3 4] -> [[1 2] [3 4] [5 6] [7 8] [9 10]]
	验证索引: [5] -> [[11 12]]
	```

	**关键观察**：
	- 训练集大小逐折增加：2 → 4 → 5
	- 验证集大小逐折减少（因为数据总量固定）
	- **所有训练索引 < 所有验证索引**（时间顺序保证）

	---

	## 6. 参数详解与效果对比

	### 6.1 test_size 固定验证集大小
	```python
	tscv = TimeSeriesSplit(n_splits=3, test_size=2)

	# 数据共6个样本，n_splits=3，test_size=2
	# 折1: 训练 [0,1], 验证 [2,3]
	# 折2: 训练 [0,1,2,3], 验证 [4,5]
	# 折3: 训练 [0,1,2,3,4], 验证 [5] ？ 注意这里只有1个了！
	# 实际上：当剩余数据不足时，会自动调整，可能产生警告
	```

	**实际运行**：
	```python
	tscv = TimeSeriesSplit(n_splits=3, test_size=2)
	for train, test in tscv.split(X):
		print(len(train), len(test))

	# 输出：
	# 2 2
	# 4 2
	# 4 2   # 第3折实际变成训练[0:4], 验证[4:6]
	```
	> ⚠️ 注意：`test_size` 固定时，如果数据不够，实际划分可能不符合预期，需计算好数据量。

	### 6.2 max_train_size 限制训练集大小
	```python
	tscv = TimeSeriesSplit(n_splits=4, max_train_size=3, test_size=2)
	# 数据12个样本
	# 折1: 训练 [0:3], 验证 [3:5]
	# 折2: 训练 [0:3], 验证 [5:7]  (仍只用最近3个)
	# 折3: 训练 [0:3], 验证 [7:9]
	# ...
	```
	**用途**：当历史数据太长，模型可能更关注近期模式，限制训练窗口可避免过拟合旧模式。

	---

	## 7. 在模型训练中实际使用

	### 手动循环训练
	```python
	from sklearn.linear_model import LinearRegression
	from sklearn.metrics import mean_squared_error
	import numpy as np

	X = np.arange(100).reshape(-1, 1)
	y = np.arange(100) + np.random.randn(100) * 5

	tscv = TimeSeriesSplit(n_splits=5)
	mse_scores = []

	for train_idx, val_idx in tscv.split(X):
		X_train, X_val = X[train_idx], X[val_idx]
		y_train, y_val = y[train_idx], y[val_idx]
		
		model = LinearRegression()
		model.fit(X_train, y_train)
		pred = model.predict(X_val)
		mse_scores.append(mean_squared_error(y_val, pred))

	print(f"平均MSE: {np.mean(mse_scores):.4f}")
	```

	### 配合 cross_val_score（需注意）
	```python
	from sklearn.model_selection import cross_val_score

	# 直接使用会报错或警告，因为 cross_val_score 默认 shuffle=False
	# 但 TimeSeriesSplit 本身不 shuffle，可以这样用：
	scores = cross_val_score(
		LinearRegression(), 
		X, y, 
		cv=TimeSeriesSplit(n_splits=5),
		scoring='neg_mean_squared_error'
	)
	print(scores.mean())
	```
	> 注意：`cross_val_score` 默认不 shuffle，所以可以配合使用，但需要确保模型支持索引切片。

	---

	## 8. TimeSeriesSplit vs KFold 对比

	| 特性 | TimeSeriesSplit | KFold |
	|------|----------------|-------|
	| **时间顺序** | 严格保持 | 忽略（可 shuffle） |
	| **训练集大小** | 逐折扩展 | 固定 = (K-1)/K * N |
	| **验证集大小** | 逐折减少（或固定） | 固定 = N/K |
	| **数据泄露风险** | 无（只用过去预测未来） | 高（可能用未来预测过去） |
	| **适用场景** | 时间序列预测 | 独立同分布数据 |
	| **shuffle 参数** | 无（不允许打乱） | 有 |

	---

	## 9. 重要注意事项与常见陷阱

	### ⚠️ 陷阱1：数据量要求
	```python
	# 如果数据太少，实际折数可能小于 n_splits
	tscv = TimeSeriesSplit(n_splits=5)
	X_small = np.arange(10).reshape(-1,1)
	print(tscv.get_n_splits(X_small))  # 可能输出 3 或更少
	```
	**原因**：每折至少需要 1 个样本做验证，且训练集至少 1 个样本。

	### ⚠️ 陷阱2：验证集大小不一致
	默认情况下，验证集大小会随着折数增加而减小（因为训练集不断扩展）。  
	**解决方法**：使用 `test_size` 固定验证集大小。

	### ⚠️ 陷阱3：未来信息泄漏（其他形式）
	- 标准化/归一化时，**必须**在训练集上 fit，再 transform 验证集（不能用全局统计量）
	- 特征工程中若用到未来时间点的信息（如未来均值），也会泄漏

	**正确做法**：在循环内分别 fit 和 transform：
	```python
	from sklearn.preprocessing import StandardScaler

	for train_idx, val_idx in tscv.split(X):
		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X[train_idx])
		X_val_scaled = scaler.transform(X[val_idx])
		# 训练模型...
	```

	### ⚠️ 陷阱4：时间间隔不均匀
	`TimeSeriesSplit` 只按**索引顺序**划分，不关心时间间隔。如果数据采集间隔不均，需自行处理或使用自定义划分。

	---

	## 10. 进阶：自定义滑动窗口（扩展）

	`TimeSeriesSplit` 只支持扩展窗口，若需要**滑动窗口**（固定长度，逐折滑动），可以自己实现或使用第三方库（如 `tslearn`）：

	```python
	# 简单滑动窗口实现示例（非 sklearn 官方）
	def sliding_window_split(X, window_size, step_size):
		n = len(X)
		for start in range(0, n - window_size, step_size):
			train_end = start + window_size
			val_end = min(train_end + step_size, n)
			yield (range(start, train_end), range(train_end, val_end))
	```

	---

	## 11. 总结与最佳实践

	| 场景 | 建议 |
	|------|------|
	| **时间序列预测** | 必须用 `TimeSeriesSplit` |
	| **数据量大（>10k）** | 设置 `max_train_size` 限制训练集大小，加速训练 |
	| **需要固定验证集大小** | 设置 `test_size` |
	| **需要更稳定的评估** | 多次运行不同起始点（手动实现前向验证） |
	| **类别不均衡的时间序列** | 结合 `StratifiedKFold` 无法使用（时间序列不能打乱），需用其他方法（如分层采样但保持顺序） |

	---

	## 12. 一句话总结

	> **TimeSeriesSplit 是 sklearn 中专门用于时间序列的交叉验证工具，采用扩展窗口策略，严格保证训练集在验证集之前，避免用未来预测过去；使用时需注意数据量、验证集大小一致性以及预处理中的数据泄露问题。**

	---

	如果你还需要了解：
	- **如何与 GridSearchCV 配合调参**
	- **滑动窗口与扩展窗口的优劣对比**
	- **时间序列预测中的嵌套交叉验证（避免调参时泄露）**

	欢迎继续提问，我可以深入展开！

* 详细讲讲 sklearn 中的 ShuffleSplit

	`ShuffleSplit` 是 scikit-learn 中一个非常灵活的交叉验证器。它的核心逻辑是：**在每次迭代中，独立地、随机地打乱整个数据集，然后按照你指定的比例（或数量）抽取样本，构成一个训练集和一个测试集**。

	这个过程会重复 `n_splits` 次，从而生成多组不同的划分。

	### 📖 核心参数解读
	使用 `ShuffleSplit` 时，主要关注以下几个参数，它们决定了划分的具体方式。

	| 参数 | 说明 | 默认值 |
	| :--- | :--- | :--- |
	| **`n_splits`** | 重新洗牌和划分的迭代次数。也就是总共会生成多少组不同的训练/测试集划分。 | `10` |
	| **`test_size`** | 测试集的大小。可以是一个浮点数（表示占总数据集的比例，如 `0.2` 代表20%）或一个整数（表示样本的绝对数量）。 | `None` |
	| **`train_size`** | 训练集的大小。用法和 `test_size` 一样。如果 `train_size` 和 `test_size` 都设为 `None`，则测试集大小默认为 `0.1`，其余作为训练集。 | `None` |
	| **`random_state`** | 随机数种子。设置一个整数可以保证每次运行代码时，产生的随机划分结果都是一样的，这对于实验的可复现性至关重要。 | `None` |

	### ⚙️ 它是如何工作的？
	你可以把 `ShuffleSplit` 的工作流程理解为以下两步的简单循环：

	1.  **随机打乱**：将整个数据集的索引完全打乱。
	2.  **按比例切分**：根据 `train_size` 和 `test_size` 的参数，从打乱后的索引中顺序取出对应数量的样本，分别作为训练集和测试集的索引。

	### 🆚 与 KFold 的核心区别
	理解 `ShuffleSplit`，最关键的就是把它和传统的 `KFold` 区分开。

	*   **`KFold` (含 `shuffle=True`)**：会将数据集**一次性**划分为 `k` 个互不重叠的“折叠”（folds）。在 `k` 次迭代中，每次都选择其中一个不同的折叠作为测试集，其余 `k-1` 个折叠作为训练集。因此，**每个样本在测试集中只会出现一次**，所有测试集是互斥的。

	*   **`ShuffleSplit`**：每次迭代都是**独立的随机抽样**。这意味着，同一个样本在不同次的迭代中，**可能被多次选入测试集，也可能一次都没有被选入**。

	这个特性直接导致了它们适用场景的差异：
	*   使用 `KFold` 进行交叉验证，能确保评估结果的方差较低，是更“标准”的做法。
	*   由于 `ShuffleSplit` 允许测试集重叠，其评估结果可能**不够稳健**。因此，一个普遍的建议是，当使用 `ShuffleSplit` 时，**设置一个比 `KFold` 更大的 `n_splits` 值**，以通过更多的随机抽样来获得更可靠的性能估计。

	### 📊 直观对比：KFold vs. ShuffleSplit
	下面的例子可以更清晰地展示这种区别。假设我们有10个样本（编号0-9），进行5次划分：

	*   **`KFold` (n_splits=5)**：每个测试集包含2个**互不重叠**的样本。
		*   第1次测试: `[1, 8]`
		*   第2次测试: `[4, 6]`
		*   第3次测试: `[2, 7]`
		*   第4次测试: `[0, 5]`
		*   第5次测试: `[3, 9]`

	*   **`ShuffleSplit` (n_splits=5, test_size=2)**：每个测试集是**随机抽取**的2个样本，会出现重叠。
		*   第1次测试: `[3, 9]`
		*   第2次测试: `[8, 2]`
		*   第3次测试: `[3, 7]` (注意，样本3再次出现)
		*   第4次测试: `[9, 0]` (注意，样本9再次出现)
		*   第5次测试: `[1, 8]` (注意，样本8再次出现)

	### 💡 使用示例
	下面是一个简单的代码示例，演示了如何使用 `ShuffleSplit`：

	```python
	from sklearn.model_selection import ShuffleSplit
	import numpy as np

	# 创建一些示例数据
	X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
	y = np.array([1, 2, 1, 2, 1, 2])

	# 初始化 ShuffleSplit
	# 生成 3 次划分，测试集占 25%，并固定随机种子以保证结果可复现
	ss = ShuffleSplit(n_splits=3, test_size=0.25, random_state=42)

	# 迭代查看每次生成的训练/测试集索引
	for i, (train_index, test_index) in enumerate(ss.split(X, y)):
		print(f"第 {i+1} 次划分:")
		print(f"  训练集索引: {train_index}")
		print(f"  测试集索引: {test_index}")
	```

	### 🎯 总结与使用建议
	`ShuffleSplit` 的灵活性是它最大的优点。你可以自由控制每次迭代中训练集和测试集的大小（无论是按比例还是按绝对数量），这在某些场景下非常有用。

	不过，由于其随机抽样的性质（测试集可能重叠），**它的评估结果方差会比标准的 `KFold` 更大**。当你需要一个更稳定、更可靠的模型性能评估时，`KFold` 通常是更稳妥的选择。但在一些需要灵活控制数据集划分比例，或者数据量极大、允许测试集重叠的场景下，`ShuffleSplit` 提供了一个便捷的解决方案。

* 详细讲讲 sklearn 中的参数搜索

	在`sklearn`中，参数搜索是选取模型最佳超参数的核心手段。它的基本思想是为一个评估器（如`SVC`）定义一个参数空间，并借助交叉验证（Cross-Validation）来系统性地评估不同参数组合的性能，最终找出评分最高的那一组。

	`sklearn`主要提供了三种策略，适用于不同的场景和需求。

	### 1. 详尽网格搜索 (`GridSearchCV`)
	这种方法的原理是“穷举”。你需要为每个超参数指定一个**候选值列表**，`GridSearchCV`会像打表一样，尝试所有可能的参数组合。

	*   **核心参数**：`param_grid`。参数值是一个字典或字典列表，字典的键是参数名，值是要尝试的数值列表。
	*   **使用示例**：如下代码会组合出 `{'kernel':'linear', 'C':1}`、`{'kernel':'rbf', 'C':10, 'gamma':0.001}` 等多种组合，进行交叉验证评估。
		```python
		param_grid = [
		  {'C': [1, 10, 100], 'kernel': ['linear']},
		  {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
		]
		```

	*   **优缺点**：优点是简单可靠，只要参数空间定义好，一定能找到全局最优组合；缺点是**计算成本极高**，当参数多或每个参数的候选值多时，组合数会呈指数级爆炸。

	### 2. 随机参数搜索 (`RandomizedSearchCV`)
	这是对网格搜索的改进，原理是“随机采样”。你为每个参数指定一个**分布**（如指数分布、均匀分布），它会在给定的迭代次数（`n_iter`）内，从这些分布中随机抽取参数组合进行评估。

	*   **核心参数**：`param_distributions` 和 `n_iter`。`param_distributions`定义参数的分布空间，`n_iter`控制采样次数。
	*   **使用示例**：如下代码会从`C`的指数分布和`gamma`的指数分布中随机采样10组参数。对于连续型参数，**强烈推荐使用连续分布**（如`scipy.stats`中的`uniform`、`loguniform`），这样才能充分利用随机搜索的优势。
		```python
		from scipy.stats import expon, loguniform
		param_dist = {
			'C': expon(scale=100),
			'gamma': expon(scale=.1),
			'kernel': ['rbf']
		}
		# n_iter=10 表示随机尝试10组参数
		```

	*   **优缺点**：优点是**效率极高**，可以在有限的计算预算内探索非常大的参数空间，并且增加不重要的参数（理论上）不会降低搜索效率；缺点是结果有一定随机性，不能保证找到绝对的全局最优。

	### 3. 连续减半搜索 (`HalvingGridSearchCV` & `HalvingRandomSearchCV`)
	这是一种更高效的迭代式搜索策略，原理是“锦标赛制”。在迭代初期，它会用少量的资源（如训练样本数）评估所有候选参数，然后只保留表现**靠前的一半**候选者，并在下一轮分配更多资源给这些幸存者，如此反复，直到选出最终的胜者。

	*   **核心机制**：通过参数`factor`（通常设为3）控制每轮资源和候选者数量的变化倍数，用`min_resources`控制初始资源量。这种机制能极大加速搜索过程，因为它能快速淘汰表现不佳的候选者。
	*   **重要提示**：这些类在`sklearn`中目前仍属于**实验性**功能，使用时需要显式导入 `from sklearn.experimental import enable_halving_search_cv`。

	### 实战中的关键技巧

	无论使用哪种搜索策略，结合以下几点能让你事半功倍：

	1.  **为模型选择正确的指标**：默认使用评估器的`score`方法（分类用准确率，回归用R²）。但在不平衡分类等问题中，准确率可能失效，此时需通过`scoring`参数指定更合适的指标（如`'f1'`、`'roc_auc'`）。
	2.  **与管道（`Pipeline`）结合**：参数搜索能完美配合`Pipeline`，甚至可以对不同组件（如特征提取器、降维器、分类器）的参数进行联合调优。
	3.  **防止数据泄露**：**务必**将原始数据先划分为一个**开发集（用于搜索和调优）**和一个**独立的测试集（用于最终评估）**，且测试集在整个调优过程中都不能被“看到”。
	4.  **并行计算与容错**：使用`n_jobs=-1`可以并行运行所有候选组合的计算，大幅提升速度。此外，若担心某些参数组合会导致模型拟合失败，可以设置`error_score=0`或`np.nan`，使搜索过程能跳过错误，继续进行。

	### 汇总与选择指南

	下表简洁对比了这三种主流方法，你可以根据实际情况选择：

	| 特性 | GridSearchCV (网格搜索) | RandomizedSearchCV (随机搜索) | Halving Search (连续减半) |
	| :--- | :--- | :--- | :--- |
	| **搜索策略** | 穷举所有组合 | 随机采样指定数量的组合 | 迭代式筛选，逐轮淘汰 |
	| **核心优势** | 结果确定，能找到最优 | 效率高，适合大空间 | 极快，计算资源利用率最高 |
	| **适用场景** | 参数空间小，组合数有限 | 参数空间大，有计算预算限制 | 资源紧张，追求极致速度 |
	| **主要缺点** | 计算成本极高 | 结果有一定随机性 | 目前是实验性功能，机制稍复杂 |

	总结来说，**如果参数组合少于100种，网格搜索简单可靠；如果参数组合成百上千，优先选择随机搜索；如果追求极致效率且不介意使用实验功能，可以尝试连续减半搜索。**

* 详细讲讲 GridSearchCV

	好的，我们来深入剖析 `sklearn` 中最经典、最核心的参数搜索工具——**`GridSearchCV`**（详尽网格搜索）。

	如果说随机搜索是“游击战”，那 `GridSearchCV` 就是一场“地毯式轰炸”。它的哲学非常简单粗暴：**既然我不知道哪个参数最好，那我就把所有可能的参数组合都试一遍，然后选出最好的那个。**

	下面我从底层原理、核心参数、进阶用法到实战避坑，为你详细拆解。

	---

	### 1. 核心工作原理（暴力美学）

	`GridSearchCV` 的执行流程可以分为三个步骤：

	1. **构建参数网格**：你传入一个字典（或字典列表），定义每个超参数的候选值。例如 `{'C': [0.1, 1, 10], 'gamma': [0.01, 0.1]}`。
	2. **笛卡尔积组合**：它会计算所有参数的笛卡尔积。上述例子会产生 `3 * 2 = 6` 种组合（`(0.1,0.01)`, `(0.1,0.1)`, `(1,0.01)`...）。
	3. **交叉验证评估**：针对**每一种**组合，它都会在训练数据上执行完整的 **K 折交叉验证**（默认 5 折）。计算每一折的验证集得分，取平均值，作为该组参数的性能指标。
	4. **选出最优**：比较所有组合的平均交叉验证得分，选出得分最高的那一组参数，并自动用**全部训练数据**重新拟合一次模型（即 `refit=True` 默认行为）。

	---

	### 2. 核心参数详解（构造函数）

	当你使用 `GridSearchCV(estimator, param_grid, ...)` 时，这几个核心参数必须吃透：

	#### (1) `estimator`（评估器）
	传入一个未拟合的模型实例，如 `SVC()`、`RandomForestClassifier()`。注意，不要传入已经 `.fit()` 过的模型。

	#### (2) `param_grid`（参数网格）—— **最关键**
	它的格式非常灵活，有两种用法：

	- **单一字典**：所有参数的组合一起尝试。
	  ```python
	  param_grid = {
		  'n_estimators': [50, 100, 200],
		  'max_depth': [None, 10, 20]
	  }  # 共 3*3=9 种组合
	  ```

	- **字典列表**（进阶用法）：当某些参数组合无效，或者想针对不同模型组件设置不同参数时使用。**搜索器会依次遍历列表中的每个字典**。
	  ```python
	  param_grid = [
		  # 场景1：线性SVM，不需要 gamma 参数
		  {'kernel': ['linear'], 'C': [0.1, 1]},
		  # 场景2：RBF核，需要 gamma 参数
		  {'kernel': ['rbf'], 'C': [0.1, 1], 'gamma': [0.001, 0.01]}
	  ]
	  # 总共 1*2 + 1*2*2 = 2+4=6 种组合
	  ```

	#### (3) `scoring`（评估指标）
	默认使用评估器自带的 `.score()` 方法（分类默认准确率，回归默认 R²）。但在实际业务中，建议显式指定，例如：
	- 分类不平衡：`scoring='f1_macro'` 或 `'roc_auc'`
	- 回归：`scoring='neg_mean_squared_error'`（注意是负值，越大越好）
	- 可以传入自定义函数，甚至多个指标（需要用 `refit` 指定以哪个为准）。

	#### (4) `cv`（交叉验证折数）
	- 传入整数（如 `cv=5`）表示 5 折分层 K 折。
	- 也可以传入交叉验证生成器（如 `StratifiedKFold`），用于更精细的控制，比如时间序列数据用 `TimeSeriesSplit`。

	#### (5) `refit`（重训练）—— **默认极重要**
	默认 `refit=True`。意思是：找到最佳参数后，**自动使用全部训练数据**（而非交叉验证的那部分）重新训练一个最终模型。这个最终模型存储在 `grid_search.best_estimator_` 中，可以直接用于 `.predict()`。

	#### (6) `n_jobs`（并行计算）
	`n_jobs=-1` 表示使用所有 CPU 核心。**注意**：并行计算会消耗大量内存，因为每个核心都要复制一份数据集。

	---

	### 3. 重要的属性（搜索完成后获取结果）

	训练完 `grid_search.fit(X_train, y_train)` 后，这些属性是你最常调用的：

	| 属性 | 含义 |
	| :--- | :--- |
	| `best_params_` | **最优参数组合**（字典形式）。这是你最想要的结果。 |
	| `best_score_` | 最优参数在交叉验证上的**平均得分**。 |
	| `best_estimator_` | 用最优参数在**全量训练集**上重新训练的模型（可直接预测）。 |
	| `cv_results_` | **极其详细**的 DataFrame（实际是字典），包含每一组参数的训练时间、每折得分、平均分、标准差等。建议转为 `pd.DataFrame(grid_search.cv_results_)` 查看。 |
	| `best_index_` | 最优参数在 `cv_results_` 中的索引位置。 |

	---

	### 4. 进阶实战技巧

	#### (1) 与 Pipeline 无缝结合
	这是实战中最常用的套路。你可以同时搜索**特征工程步骤**和**模型**的超参数。参数名用 `{步骤名}__{参数名}` 的格式。

	```python
	from sklearn.pipeline import Pipeline
	from sklearn.decomposition import PCA
	from sklearn.svm import SVC

	pipe = Pipeline([('pca', PCA()), ('svc', SVC())])

	param_grid = {
		'pca__n_components': [10, 20, None],  # 搜索降维维度
		'svc__C': [0.1, 1, 10],
		'svc__kernel': ['linear', 'rbf']
	}
	# GridSearchCV 会对整个管道进行搜索，确保交叉验证时不泄露数据。
	```

	#### (2) 自定义评分函数
	如果你的业务需要个性化打分（比如“精准率必须大于 0.9 的前提下最大化召回率”），可以用 `make_scorer` 封装：

	```python
	from sklearn.metrics import make_scorer, f1_score
	grid = GridSearchCV(estimator, param_grid, 
						scoring=make_scorer(f1_score, average='macro'))
	```

	#### (3) 提前剔除无效组合（`ParameterGrid` 的妙用）
	如果某些参数组合在逻辑上是无效的（例如 SVM 的 `kernel='linear'` 时不需要 `gamma`），你可以使用**字典列表**来规避，而不是把 `gamma` 设为 `[0.01, None]` 这种冗余写法。

	---

	### 5. 致命缺点与应对策略

	| 问题 | 说明 | 应对方案 |
	| :--- | :--- | :--- |
	| **维度灾难** | 如果有 10 个参数，每个参数 10 个候选值，组合数为 `10^10`，根本无法计算。 | 先使用 `RandomizedSearchCV` 进行粗筛，缩小参数范围；然后再用 `GridSearchCV` 在小范围内精细搜索。 |
	| **资源浪费** | 如果一组参数在某折验证中表现极差，网格搜索仍会继续跑完剩下的所有折数。 | 可以换用 `HalvingGridSearchCV`，它会提前淘汰表现差的候选者。 |
	| **过拟合风险** | 搜索的候选值越多，理论上越可能“碰巧”选到在验证集上表现好但在测试集上差的参数。 | **严格遵循**：先分测试集（不参与搜索），仅在训练集上做 GridSearch，防止信息泄露。 |

	---

	### 6. 完整代码实战示例

	```python
	from sklearn.model_selection import GridSearchCV
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.datasets import make_classification
	from sklearn.model_selection import train_test_split

	# 1. 准备数据
	X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# 2. 定义模型和参数网格
	model = RandomForestClassifier(random_state=42)
	param_grid = {
		'n_estimators': [50, 100, 200],
		'max_depth': [None, 10, 20],
		'min_samples_split': [2, 5, 10]
	}
	# 总共 3*3*3 = 27 种组合，每种组合做 5 折 CV => 135 次拟合

	# 3. 执行网格搜索（并行+显式评分）
	grid_search = GridSearchCV(
		estimator=model,
		param_grid=param_grid,
		cv=5,
		scoring='f1_macro',       # 针对多分类使用宏平均F1
		n_jobs=-1,                # 使用所有CPU核心
		verbose=1                 # 打印进度条
	)
	grid_search.fit(X_train, y_train)

	# 4. 查看结果
	print("最佳参数：", grid_search.best_params_)
	print("最佳交叉验证得分：", grid_search.best_score_)

	# 5. 在测试集上评估（这才是最终泛化能力）
	best_model = grid_search.best_estimator_
	test_score = best_model.score(X_test, y_test)
	print("测试集得分：", test_score)

	# 6. （调试）查看所有参数组合的详细结果
	results = pd.DataFrame(grid_search.cv_results_)
	# 按测试得分降序排列，查看前5组
	print(results[['params', 'mean_test_score', 'std_test_score']].sort_values('mean_test_score', ascending=False).head())
	```

	---

	### 总结一句
	`GridSearchCV` 是你学习机器学习的**基本功**。它的原理简单可靠，是模型调优的“金标准”。在实际工作中，我的建议是：**先用随机搜索确定“感兴趣”的参数区域，最后用网格搜索在这个小区域内进行精挑细选**，这样既保证了搜索质量，又兼顾了计算效率。

	如果你对“如何结合随机搜索做粗调”感兴趣，或者想了解自定义参数采样的高级技巧，可以随时告诉我，我再为你展开。

* `classification_report(y_gt, y_pred)` 要求两个 ndarray 都必须是 int 类型，不能是 float 类型

	如果前面使用了 `LinearRegression()` 之类的 `predict()`，会导致 `y_pred` 为 float。此时找 `y_pred` 离哪个整数更近，得到分类的结果，试了下效果还可以。

	```py
	# 在这个例子中，分类 label 为 [0, 1, 2]
	
	# 线性回归的输出其实是 float
	y_pred = lr.predict(X_test)  # float64

	# 找到离 y_pred 最近的整数。这里使用了广播机制
	y_pred = np.abs(y_pred.reshape(-1, 1) - np.array([0, 1, 2]).reshape(1, 3)).argmin(axis=1)
	```

* `train_test_split()` 拆分时间序列数据时，似乎只能按“前半段 (train) +  后半段 (test)”拆分。不能做成按 sclice / block 把时间序列拆分成小块，再对这些小块进行拆分训练集和验证集。

	或许可以先手动使用 range()，time_arr[::step] 之类的方式把时间序列拆分成小块，变成 (num_blocks, num_points)，对第一个维度进行 split，即可得到训练集和验证集。也可以对第二个维度进行 split，得到前半段 (train) + 后半段（test）的拆分。

	其实如果 `train_test_split()` 不 shuffle 的话，基本等于手动使用 train_ratio * num_points 计算拆分点了，然后使用 `train_set = arr[:train_ratio * num_points]`, `test_set = arr[train_ratio * num_points:]` 就可以了

* 详细讲讲 sklearn 中的 preprocessing 模块

	`sklearn.preprocessing` 是 scikit-learn 中用于数据预处理的模块，它包含了许多实用的函数和转换器类，主要目标是把原始特征向量转换成更适合机器学习模型的形式。

	简单来说，它的核心价值在于：**许多机器学习算法（如线性模型、SVM等）都默认数据是服从标准正态分布或特征处于同一尺度。如果原始数据不满足这个条件，模型效果可能会大打折扣。** `preprocessing` 模块就是为解决这个问题而生的。

	这个模块的功能主要可以分为以下几类，我为你整理了一个表格以便快速了解：

	| 功能类别 | 核心目标 | 常用工具 |
	| :--- | :--- | :--- |
	| **标准化与缩放** | 改变特征的数值范围，消除量纲影响 | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler` |
	| **非线性变换** | 改变数据分布，使其更接近高斯分布或均匀分布 | `QuantileTransformer`, `PowerTransformer` |
	| **归一化** | 将单个样本缩放到单位范数（长度） | `Normalizer` |
	| **编码分类特征** | 将文本或类别型数据转换为数值型 | `OneHotEncoder`, `OrdinalEncoder` |
	| **离散化与二值化** | 将连续特征分段或按阈值转为0/1 | `KBinsDiscretizer`, `Binarizer` |

	### ⚖️ 标准化与缩放 (Standardization & Scaling)

	这是最常用的功能，主要解决不同特征之间数值尺度差异过大的问题。比如，一个特征是“年龄”（0-100），另一个是“年收入”（0-100000），如果直接使用，模型可能会忽略“年龄”的影响。

	-   **`StandardScaler` (Z-score标准化)**：它通过**减去均值，再除以标准差**，将数据转换为均值为0，方差为1的标准正态分布。这适用于大多数假设数据服从正态分布的算法，如线性回归、逻辑回归、SVM等。
	-   **`MinMaxScaler` (最大-最小缩放)**：它将数据**缩放到一个指定的范围，通常是 [0, 1]**。其计算方式为 `(X - X.min()) / (X.max() - X.min())`。这种缩放方式对数据边界敏感，适合需要将特征值限定在特定区间的场景，如图像像素值处理。
	-   **`RobustScaler` (稳健缩放)**：如果数据中包含大量**离群值 (outliers)**，它会是一个很好的选择。因为它使用的是**中位数和四分位数**（如IQR）等稳健统计量，而不是均值和标准差，所以受异常值影响较小。
	-   **`MaxAbsScaler` (最大绝对值缩放)**：它将每个特征除以该特征的最大绝对值，使数据缩放到 [-1, 1] 范围内。它特别适合处理**稀疏数据 (sparse data)**，因为它不会破坏数据的稀疏结构。

	**一个至关重要的原则**：这些缩放器都遵循 **"先fit，后transform"** 的模式。你只能在**训练集**上调用 `.fit()` 方法（学习均值、标准差等参数），然后在训练集和测试集上都调用 `.transform()` 方法来应用转换。这样可以确保数据不会泄露，并保证测试数据与训练数据使用相同的尺度进行转换。

	### 🧬 非线性变换 (Non-linear Transformation)

	当数据分布过于“古怪”或偏离高斯分布时，缩放可能不够，非线性变换能更好地处理这类情况。

	-   **`QuantileTransformer` (分位数变换)**：它会将特征的分布映射到指定的分布（如均匀分布或正态分布）上。它通过计算每个数据点的分位数，并将其映射到目标分布的分位数上，从而**平滑数据分布并减少离群值的影响**，但代价是可能会扭曲特征间的相关性。
	-   **`PowerTransformer` (幂变换)**：它使用如**Box-Cox**或**Yeo-Johnson**变换，通过参数化的方法将数据映射到更像高斯分布的形态，使其更符合许多统计模型的假设。

	### 📐 归一化 (Normalization)

	这里的“归一化”特指对**样本**进行操作，而非特征。`Normalizer` 会将**每个样本**（即每行数据）缩放到单位范数（欧几里得长度或L1范数为1）。这在文本分类或聚类中很常见，当样本间的相似度用**点积或余弦相似度**来衡量时，这个方法很有用。

	### 🏷️ 处理分类特征 (Encoding Categorical Features)

	原始数据中经常包含“性别”、“颜色”等类别型特征，模型无法直接处理。

	-   **`OrdinalEncoder` (序数编码)**：将类别转换为整数，如 0, 1, 2。但它会强加一个"顺序"，可能导致模型错误地认为类别间存在大小关系。
	-   **`OneHotEncoder` (独热编码)**：为每个类别创建一个新的二元特征（0或1），是处理**无序分类变量**最常用的方法，可以避免引入虚假的顺序关系。

	### 🎯 离散化与二值化 (Discretization & Binarization)

	-   **`KBinsDiscretizer`**：将连续的数值特征划分为 `k` 个离散的区间（分箱），可以将线性模型转化为非线性模型。
	-   **`Binarizer`**：根据设定的阈值，将数值特征二值化为 0 或 1。

	### 💡 如何使用

	通常，你会将这些预处理转换器与 `Pipeline`（管道）结合使用，将预处理步骤和模型训练步骤封装在一起。这样做既方便、整洁，又能确保数据预处理的正确性（例如在交叉验证时避免数据泄露）。用法如下：

	```python
	from sklearn.pipeline import make_pipeline
	from sklearn.preprocessing import StandardScaler
	from sklearn.linear_model import LogisticRegression

	# 创建一个管道：先标准化，再训练逻辑回归
	pipe = make_pipeline(StandardScaler(), LogisticRegression())
	pipe.fit(X_train, y_train)  # 训练时，自动先对数据做标准化
	pipe.score(X_test, y_test)  # 预测时，使用训练集的参数对测试数据做同样的标准化
	```

	`sklearn.preprocessing` 模块是数据科学工作流中一个非常关键的组件，你通常需要根据数据的特性和模型的假设，来选择最合适的预处理方法。

	如果你对某个具体的转换器（比如如何处理缺失值，或者更复杂的 `ColumnTransformer`）感兴趣，可以随时告诉我，我们可以继续深入探讨。

* 详细讲讲 sklearn 中的 model_selection 模块

	`sklearn.model_selection` 是 scikit-learn 中一个非常核心的模块，它就像一个为模型挑选“最优配置”和验证“真实水平”的工具箱。它将原本分散在 `cross_validation`、`grid_search` 和 `learning_curve` 等旧模块中的功能整合到了一起。

	它的核心功能主要围绕三个方面展开：**将数据划分为训练/测试集**、**通过交叉验证评估模型**，以及**自动搜索模型的最佳超参数**。

	我把它的核心功能和使用场景整理成了下面这个表格，方便你快速概览：

	| 功能类别 | 核心目标 | 常用工具 | 一句话适用场景 |
	| :--- | :--- | :--- | :--- |
	| **数据划分** | 将数据集切分为训练集和测试集 | `train_test_split` | 快速获得用于初步验证的独立测试集。 |
	| **交叉验证 (评估)** | 利用数据的不同子集多次评估模型，得到更稳健的性能指标 | `cross_val_score`, `cross_validate`, `cross_val_predict` | 评估模型泛化能力，避免一次划分带来的偶然性。 |
	| **超参数搜索** | 自动寻找使模型性能最优的参数组合 | `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV` | 当模型有很多参数需要调节，且希望自动化寻找最佳组合时。 |

	### 📊 数据划分：Train-Test Split

	这是最基础的一步，用于快速划分出一部分数据作为“留出集”，来初步测试模型的表现。

	*   **`train_test_split`**：这是最常用的函数，默认按75%/25%的比例将数据随机划分为训练集和测试集。
	*   **分层划分 (`stratify`)**：对于分类问题，特别是数据类别不平衡时，设置参数 `stratify=y` 可以确保训练集和测试集中各类别的比例与原数据集保持一致，这是非常重要的实践。

	### 🔬 交叉验证：更可靠的模型评估

	交叉验证的核心思想是重复使用数据，将数据分成 `k` 份（称为“折”），轮流将其中的一份作为验证集，其余 `k-1` 份作为训练集，最终得到 `k` 个评估结果，其平均值作为模型性能的最终估计。这能有效避免因单次数据划分不当而导致的评估偏差。

	scikit-learn提供了多种交叉验证策略，你可以把它们想象成不同的“拆分器”（Splitter），每个都通过 `split` 方法来生成训练/测试索引。

	*   **`KFold`**：最标准的K折交叉验证，将数据随机分成K份。
	*   **`StratifiedKFold`**：**分层K折**。它是 `KFold` 的变种，在每一折中都努力保持和原数据集相同的类别比例，是**分类问题首选**。
	*   **`GroupKFold`**：**分组K折**。确保来自同一组（如同一个病人的多次检测数据）的样本不会同时出现在训练集和验证集中，适用于样本不独立的情况。
	*   **`TimeSeriesSplit`**：**时间序列分割**。用于时间序列数据，它保证训练集总是在验证集的时间顺序**之前**，避免“用未来预测过去”的数据泄露问题。
	*   **`LeaveOneOut` (LOO)**：**留一法**。当数据量非常小时，可以每次只留一个样本作为验证集，计算量巨大。

	你可以通过以下几个便利函数来应用这些策略：
	*   **`cross_val_score`**：直接返回每次交叉验证的评分数组。
	*   **`cross_validate`**：功能更强大，可以同时返回多个评估指标（如准确率、精确率、召回率）以及模型的训练/测试时间。
	*   **`cross_val_predict`**：返回每个样本在交叉验证过程中被当作验证集时的预测结果，便于你后续做更细致的错误分析。

	### 🚀 超参数搜索：自动寻找最佳参数

	模型的“超参数”是我们在训练前需要手动设定的，比如支持向量机的 `C` 和 `gamma` 值。这个模块提供了自动搜索最佳超参数组合的工具。

	*   **`GridSearchCV` (网格搜索)**：**暴力但可靠**。它会穷举你指定的所有参数组合，运行交叉验证，并找出其中表现最好的一组。如果参数组合很多，这个过程会非常耗时。
	*   **`RandomizedSearchCV` (随机搜索)**：**更高效**。它不会尝试所有组合，而是从你指定的参数分布中随机采样固定数量（由 `n_iter` 控制）的组合，然后用交叉验证评估。这在参数空间很大时，往往能以更低的成本找到不错的参数。
	*   **`HalvingGridSearchCV` / `HalvingRandomSearchCV` (连续减半搜索)**：**更聪明、更快速**。它采用“连续减半”策略，先用少量数据评估所有候选组合，淘汰表现差的，再用更多数据评估剩下的优胜者，如此迭代，直到选出最佳组合。

	`GridSearchCV` 和 `RandomizedSearchCV` 等搜索器在使用上和模型本身很相似，调用 `.fit()` 方法即可开始搜索。搜索完成后，可以从它的属性中获取最佳参数组合和最佳模型：
	*   `best_params_`：最佳参数组合。
	*   `best_score_`：最佳参数组合对应的交叉验证平均得分。
	*   `best_estimator_`：使用最佳参数在整个训练集上重新训练好的模型。
	*   `cv_results_`：一个包含所有搜索过程的详细结果的字典，可以轻松导入 Pandas DataFrame 进行深入分析。

	`model_selection` 模块是连接数据预处理和模型训练的关键环节。正确地使用它，可以帮助你更好地评估模型的真实水平，并高效地进行调参。

	> **💡 进阶技巧：嵌套交叉验证 (Nested Cross-Validation)**
	> 当你使用 `GridSearchCV` 进行调参时，你实际上已经在做一次交叉验证了（为了选参数）。但如果你用调参后选出的最佳模型的表现（如 `best_score_`）来作为最终模型的性能指标，这个估计可能是**有偏乐观的**。
	> 更严谨的做法是进行**嵌套交叉验证**：外层用一个交叉验证（如 `cross_val_score`）来评估整个调参流程（`GridSearchCV` 对象本身）的泛化性能。这样，内层的交叉验证用于选参数，外层的交叉验证用于评估模型。`model_selection` 模块的设计（特别是数据独立的 CV 拆分器）正是为了支持这种高级用法。

* 详细讲讲 sklearn 中 LinearRegression() 的用法

	`LinearRegression` 是 scikit-learn 中实现**普通最小二乘法（Ordinary Least Squares, OLS）** 线性回归的类，属于`sklearn.linear_model`模块。它的目标是通过学习一个线性模型，来最小化数据集中真实目标值与模型预测值之间的**残差平方和**。

	其核心用法和关键点如下。

	### 核心使用流程

	使用`LinearRegression`通常遵循 scikit-learn 的标准三步走模式：创建、拟合、预测与评估。

	```python
	import numpy as np
	from sklearn.linear_model import LinearRegression
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import mean_squared_error, r2_score

	# 1. 准备数据
	# X 是特征矩阵 (n_samples, n_features)，y 是目标向量 (n_samples,)
	# 这里用示例数据，实际数据中 X, y 需要提前定义
	# X = ...
	# y = ...
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# 2. 创建并拟合模型
	model = LinearRegression() # 创建模型实例
	model.fit(X_train, y_train) # 使用训练数据拟合模型

	# 3. 进行预测
	y_pred = model.predict(X_test)

	# 4. 评估模型
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)
	print(f"均方误差 (MSE): {mse:.2f}")
	print(f"决定系数 (R²): {r2:.2f}")
	```

	### 关键参数详解

	在创建`LinearRegression`对象时，可以通过几个参数来控制模型的行为。

	*   **`fit_intercept` (默认 `True`)**：决定是否计算截距项。如果设为`False`，模型将强制通过原点（即`intercept_`为0），这通常意味着你事先知道数据是中心化的。
	*   **`positive` (默认 `False`)**：当设为`True`时，会强制所有特征的系数均为正数。这个选项只适用于稠密数据（非稀疏矩阵）。
	*   **`n_jobs` (默认 `None`)**：用于加速计算的CPU核心数。当目标值有多个（多输出回归）或数据是稀疏矩阵时，设置`-1`可以调用所有处理器来加快速度。
	*   **`copy_X` (默认 `True`)**：是否在拟合过程中复制特征矩阵`X`，如果设为`False`，则可能会覆盖原始数据以节省内存。
	*   **`tol` (默认 `1e-6`)**：控制解的精度的公差参数，对底层的求解器起作用。

	### 模型的重要属性

	模型拟合完成后，可以调用以下属性来查看学习到的结果。

	*   **`coef_`**：一个数组，表示模型的系数。如果`y`是一维的，`coef_`的形状是`(n_features,)`；如果`y`是二维（多目标输出），则形状为`(n_targets, n_features)`。
	*   **`intercept_`**：一个浮点数或数组，表示模型的截距。如果`fit_intercept`为`False`，则其值为`0.0`。
	*   **`rank_`**：矩阵`X`的秩，仅在数据为稠密时可用。
	*   **`singular_`**：矩阵`X`的奇异值，仅在数据为稠密时可用。

	### 进阶用法

	#### 1. 添加样本权重
	`fit`方法支持 `sample_weight` 参数，允许为不同的训练样本赋予不同的重要性。这在处理不均衡数据或需要强调特定样本时很有用。

	```python
	# sample_weight 是一个与样本数量相同的数组
	weights = np.array([0.5, 1.0, 1.2, ...]) 
	model.fit(X, y, sample_weight=weights)
	```

	#### 2. 模型评估：`score()` 方法
	`LinearRegression`内置了`score()`方法，它直接返回模型在给定测试数据上的**决定系数 $R^2$**。

	```python
	r2_score = model.score(X_test, y_test)
	print(f"模型 R² 分数: {r2_score:.2f}")
	```
	$R^2$ 的最大值为1，值越接近1，说明模型对数据的拟合程度越好。它也可能为负数，表示模型的表现甚至不如一个简单的常数模型。

	#### 3. 完整示例：从糖尿病数据集学习
	官方文档提供了一个使用糖尿病数据集构建线性回归模型的完整示例。

	这个示例展示了如何加载数据、划分训练/测试集、训练模型，并最终可视化预测结果。

	### 注意事项

	*   **与正则化模型的区别**：`LinearRegression`是基础的OLS模型，容易在特征过多时发生过拟合。如果遇到此问题，可以考虑使用加入了正则化项的`Ridge`（L2正则化）或`Lasso`（L1正则化）模型。
	*   **数据假设**：虽然`LinearRegression`本身是一个计算工具，但它背后对数据有一定的统计假设，例如特征与目标之间应存在线性关系，特征之间不应存在高度共线性等。

* 在 sklearn、numpy 和 scipy 这三个库中，都没有直接名为 head() 或 tail() 的函数。

* 简述 python sklearn 中 svm 的使用方法

    我来简述 sklearn 中 SVM 的使用方法：

    1. 导入必要的库

        ```python
        from sklearn import svm
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, accuracy_score
        ```

    2. 数据准备与预处理

        ```python
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 特征缩放（SVM对特征尺度敏感）
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        ```

    3. 创建 SVM 模型

        ```python
        # 分类任务
        # 线性核
        model = svm.SVC(kernel='linear', C=1.0)

        # RBF核（默认）
        model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')

        # 多项式核
        model = svm.SVC(kernel='poly', degree=3, C=1.0)

        # 回归任务
        model = svm.SVR(kernel='rbf', C=1.0)
        ```

    4. 训练模型

        ```python
        model.fit(X_train, y_train)
        ```

    5. 预测与评估

        ```python
        # 预测
        y_pred = model.predict(X_test)

        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        print(f"准确率: {accuracy}")
        print(classification_report(y_test, y_pred))
        ```

    6. 完整示例

        ```python
        from sklearn import svm, datasets
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # 加载数据
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # 特征缩放
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 创建并训练SVM
        clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X_train, y_train)

        # 预测
        y_pred = clf.predict(X_test)
        print(f"准确率: {accuracy_score(y_test, y_pred)}")
        ```

    关键参数说明

    - **kernel**: 核函数类型（'linear', 'rbf', 'poly', 'sigmoid'）
    - **C**: 惩罚参数，控制错误样本的惩罚力度
    - **gamma**: 核函数系数，影响单个样本的影响范围
    - **degree**: 多项式核的度数

    SVM 使用前建议进行特征缩放，并根据数据特点选择合适的核函数。

* svm 对向量形式的特征进行分类时，向量的维度设置为多少以下 svm 的效果比较好？

    关于 SVM 对向量维度的要求，没有一个固定的阈值，但可以根据经验和理论给出一些参考：

    维度建议范围

    1. **低维到中维（< 1000 维）**

        - **效果最好**：特别是样本量适中时
        - 线性可分或使用 RBF 核都能表现良好

    2. **中维到高维（1000 - 10^5 维）**

        - **依然有效**：但需要注意：

            - 计算复杂度增加

            - 过拟合风险增加

            - 建议使用**线性核**

    3. **超高维（> 10^5 维）**

        - **挑战较大**：

            - 计算开销大

            - 内存消耗大

            - 可能不如其他算法（如线性模型）

    **维度与样本量的关系**

    更重要是考虑 **维度(d) 与 样本量(n) 的比例**：

    ```python
    # 经验法则
    if n/d > 10:  # 样本充足
        # 可以使用 RBF 核等复杂核函数
        model = svm.SVC(kernel='rbf')
        
    elif n/d < 1:  # 样本少于维度（高维小样本）
        # 建议使用线性核
        model = svm.LinearSVC()  # 或 svm.SVC(kernel='linear')
    ```

    实际建议

    ```python
    from sklearn import svm

    def suggest_svm(X, y):
        n_samples, n_features = X.shape
        
        if n_features < 100:
            # 低维：可以使用复杂核函数
            return svm.SVC(kernel='rbf', C=1.0)
        
        elif n_features < 10000:
            # 中维：根据样本量决定
            if n_samples > n_features * 10:
                return svm.SVC(kernel='rbf', gamma='scale')
            else:
                return svm.LinearSVC()
        
        else:
            # 高维：建议线性核或降维
            from sklearn.decomposition import PCA
            # 先降维再使用 SVM
            pca = PCA(n_components=min(100, n_samples))
            X_reduced = pca.fit_transform(X)
            return svm.SVC(kernel='rbf')
    ```

    特殊情况

    - **文本分类**：即使维度很高（如 TF-IDF 特征 > 10^5），线性 SVM 仍然表现很好
    - **图像识别**：原始像素维度高，通常需要 CNN 或降维后再用 SVM
    - **基因数据**：维度高样本少，使用线性核 + 特征选择

    总结

    没有绝对的"效果比较好"的维度限制，关键在于：

    1. 维度与样本量的比例
    2. 数据的分布特性
    3. 选择的核函数
    4. 计算资源限制

    **最佳实践**：先尝试线性 SVM，如果效果不理想且计算资源允许，再尝试 RBF 核并做特征选择或降维。

* 详细讲讲 SVC() 的用法

    好的，我们来详细讲解一下 `sklearn.svm.SVC()`（Support Vector Classifier，支持向量分类器）。

    `SVC` 是 scikit-learn 中支持向量机（SVM）的分类实现，它基于 **libsvm** 库，是一个非常强大且经典的分类算法。

    1. 核心原理

        SVC 的目标是找到一个**超平面**，使得不同类别的样本之间的**间隔最大化**。

        - **线性可分**：找到硬间隔超平面
        - **线性不可分**：引入**软间隔**（允许部分样本误分类），通过惩罚参数 `C` 控制
        - **非线性**：通过**核函数**（Kernel）将数据映射到高维空间，使其线性可分

    2. 基本用法

        ```python
        from sklearn.svm import SVC
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # 加载数据
        iris = load_iris()
        X, y = iris.data, iris.target

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 标准化（SVM 对尺度敏感，必须做）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 创建 SVC 模型（默认参数）
        svc = SVC()

        # 训练
        svc.fit(X_train_scaled, y_train)

        # 预测
        y_pred = svc.predict(X_test_scaled)

        # 评估
        print("准确率:", svc.score(X_test_scaled, y_test))
        ```

    3. 核心参数详解

        1. C（惩罚参数, Penalty Parameter, or Regularization Parameter）

            - **默认值**: 1.0
            - **作用**: 控制对误分类样本的惩罚力度。
            - **C 越大**：对误分类惩罚越重，决策边界更复杂，可能过拟合（硬间隔）
            - **C 越小**：允许更多误分类，决策边界更平滑，可能欠拟合（软间隔）
            - **调参建议**: 通常使用网格搜索（GridSearchCV）在对数尺度上搜索，如 `[0.001, 0.01, 0.1, 1, 10, 100]`

            ```python
            # C 的作用示例
            svc_hard = SVC(C=100)   # 强硬分类
            svc_soft = SVC(C=0.01)  # 宽松分类
            ```

        2. kernel（核函数）

            - **默认值**: 'rbf'
            - **作用**: 定义如何将数据映射到高维空间，以处理非线性问题。
            - **可选值**:

            | 核函数 | 参数 | 适用场景 |
            |--------|------|----------|
            | `'linear'` | 无 | 线性可分数据，特征数多 |
            | `'poly'` | `degree`（多项式次数） | 多项式决策边界 |
            | `'rbf'` | `gamma`（高斯核宽度） | 最常用，适合大多数非线性问题 |
            | `'sigmoid'` | `gamma`, `coef0` | 类似神经网络的激活函数 |
            | 自定义核函数 | 传入函数 | 特殊需求 |

            ```python
            svc_linear = SVC(kernel='linear')
            svc_poly = SVC(kernel='poly', degree=3)
            svc_rbf = SVC(kernel='rbf')
            ```

        3. gamma（RBF/Poly/Sigmoid 核参数）**

            - **默认值**: `'scale'`（在 scikit-learn 0.22+ 中）
            - **作用**: 控制单个训练样本的影响范围。
            - **gamma 越大**：每个样本影响范围小，决策边界更复杂、更扭曲，容易过拟合
            - **gamma 越小**：每个样本影响范围大，决策边界更平滑，偏向欠拟合
            - **可选值**:
            - `'scale'`: `1 / (n_features * X.var())`（自动计算）
            - `'auto'`: `1 / n_features`
            - 正浮点数：手动指定

            ```python
            svc_gamma_large = SVC(gamma=10)   # 过拟合风险高
            svc_gamma_small = SVC(gamma=0.01) # 欠拟合风险高
            ```

        4. degree（多项式核次数）

            - **默认值**: 3
            - **作用**: 仅当 `kernel='poly'` 时有效，指定多项式的次数。

            ```python
            svc_poly_degree5 = SVC(kernel='poly', degree=5)
            ```

        5. coef0（独立项系数）

            - **默认值**: 0.0
            - **作用**: 仅对 `'poly'` 和 `'sigmoid'` 核有效，影响高维映射的偏移。

            ```python
            svc_poly = SVC(kernel='poly', coef0=1)
            ```

        6. shrinking（启发式收缩）

            - **默认值**: True
            - **作用**: 是否使用收缩启发式方法来加速训练。通常保持默认 True。

        7. probability（概率估计）

            - **默认值**: False
            - **作用**: 是否启用概率预测（`predict_proba` 方法）。
            - 启用后会进行**Platt 缩放**，增加训练时间，但可以输出类别概率。
            - 适用于需要概率输出的场景（如排序、风险评估）。

            ```python
            svc_prob = SVC(probability=True)
            svc_prob.fit(X_train, y_train)
            probs = svc_prob.predict_proba(X_test)  # 返回每个类别的概率
            ```

        8. class_weight（类别权重）
            
            - **默认值**: None
            - **作用**: 处理类别不平衡问题。
            - `None`: 所有类别权重相等
            - `'balanced'`: 自动根据样本数调整权重，`n_samples / (n_classes * np.bincount(y))`
            - 字典: 手动指定，如 `{0: 1, 1: 10}`

            ```python
            # 处理不平衡数据
            svc_balanced = SVC(class_weight='balanced')
            ```

        9. random_state（随机种子）

            - **默认值**: None
            - **作用**: 控制随机数生成（影响数据打乱等），用于结果可复现。

            ```python
            svc = SVC(random_state=42)
            ```

    4. 重要属性（Attributes）

        | 属性 | 说明 |
        |------|------|
        | `support_` | 支持向量的索引 |
        | `support_vectors_` | 支持向量的样本值 |
        | `n_support_` | 每个类别的支持向量数量 |
        | `dual_coef_` | 对偶系数（支持向量的权重） |
        | `coef_` | 仅在线性核时有效，特征权重（可用于特征重要性） |
        | `intercept_` | 决策函数的截距（偏置） |
        | `classes_` | 类别标签 |
        | `fit_status_` | 拟合状态（0 表示成功，1 表示失败） |

        ```python
        print("支持向量数量:", svc.n_support_)
        print("支持向量:", svc.support_vectors_)
        ```

    5. 重要方法（Methods）

        | 方法 | 说明 |
        |------|------|
        | `fit(X, y)` | 训练模型 |
        | `predict(X)` | 预测类别 |
        | `predict_proba(X)` | 预测类别概率（需 `probability=True`） |
        | `predict_log_proba(X)` | 对数概率 |
        | `decision_function(X)` | 决策函数值（超平面距离），可用于置信度 |
        | `score(X, y)` | 返回平均准确率 |
        | `set_params(**params)` | 设置参数 |
        | `get_params()` | 获取参数 |

        ```python
        # 决策函数（距离超平面的距离）
        decision_vals = svc.decision_function(X_test)
        print("决策值:", decision_vals[:5])
        ```

    6. 数据预处理（重要！）

        **SVM 对特征尺度极其敏感**，因为核函数依赖欧氏距离。必须进行标准化或归一化：

        ```python
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svc = SVC()
        svc.fit(X_train_scaled, y_train)
        ```

        **不标准化的后果**：特征值大的特征主导距离计算，模型性能大幅下降。

    7. 调参策略（GridSearchCV）

        SVC 有多个超参数，通常使用网格搜索寻找最优组合：

        ```python
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'linear']
        }

        svc = SVC(random_state=42)
        grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        print("最佳参数:", grid_search.best_params_)
        print("最佳交叉验证得分:", grid_search.best_score_)

        # 使用最佳模型预测
        best_svc = grid_search.best_estimator_
        y_pred = best_svc.predict(X_test_scaled)
        ```

    8. 核函数选择指南

        | 数据特点 | 推荐核函数 | 原因 |
        |---------|-----------|------|
        | 特征数 > 样本数 | `linear` | 避免过拟合，计算快 |
        | 特征数 < 样本数 | `rbf` | 处理非线性能力强 |
        | 特征数 ≈ 样本数 | `rbf` | 默认首选 |
        | 线性可分 | `linear` | 简单高效 |
        | 需要解释特征重要性 | `linear` | 可以看 `coef_` |
        | 文本分类（高维稀疏） | `linear` | 线性核效果通常最好 |
        | 图像、复杂模式 | `rbf` 或 `poly` | 高维映射 |

    9. 多分类策略

        SVC 原生支持多分类，使用 **一对一**（One-vs-One, OvO）策略：

        - 对于 K 个类别，训练 K*(K-1)/2 个二分类器
        - 每个分类器区分两个类别
        - 最终通过投票决定类别

        也可以通过 `decision_function_shape` 参数调整：

        - `'ovr'`（One-vs-Rest）：训练 K 个分类器，每个类别 vs 其余
        - `'ovo'`（默认）：一对一

        ```python
        svc_ovr = SVC(decision_function_shape='ovr')
        ```

    10. 优缺点

        ✅ 优点

        - 有效处理高维数据
        - 核技巧使其能处理非线性问题
        - 基于支持向量，内存效率高（只依赖支持向量）
        - 理论保证（VC 维理论）

        ❌ 缺点

        - 对大规模数据（> 10万样本）训练慢（O(n²) 或 O(n³)）
        - 对参数和核函数选择敏感
        - 不直接输出概率（需要 Platt 缩放）
        - 对缺失数据敏感

    11. 大规模数据替代方案

        | 方案 | 适用场景 |
        |------|----------|
        | `LinearSVC` | 线性核，大规模数据，基于 liblinear |
        | `SGDClassifier` | 超大规模数据，在线学习 |
        | `NuSVC` | 用 ν 参数代替 C，控制支持向量数量 |

        ```python
        from sklearn.svm import LinearSVC

        linear_svc = LinearSVC(C=1.0, dual=False)  # dual=False 当样本数 > 特征数
        linear_svc.fit(X_train, y_train)
        ```

    12. 完整实战示例（带调参）

        ```python
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import classification_report, confusion_matrix

        # 生成数据
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                                n_informative=15, random_state=42)

        # 划分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 调参
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': [0.01, 0.1, 1],
            'kernel': ['rbf', 'linear']
        }

        svc = SVC(random_state=42, probability=True)
        grid = GridSearchCV(svc, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)

        # 最佳模型
        best = grid.best_estimator_
        y_pred = best.predict(X_test_scaled)
        y_proba = best.predict_proba(X_test_scaled)

        print("最佳参数:", grid.best_params_)
        print("测试集准确率:", best.score(X_test_scaled, y_test))
        print("\n分类报告:\n", classification_report(y_test, y_pred))
        print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))
        ```

    13. 总结

        | 要点 | 描述 |
        |------|------|
        | 核心作用 | 通过最大化间隔进行分类 |
        | 必须预处理 | **标准化**（StandardScaler） |
        | 关键参数 | `C`（惩罚）、`kernel`（核函数）、`gamma`（核宽度） |
        | 调参方法 | GridSearchCV 或 RandomizedSearchCV |
        | 多分类 | 默认一对一（OvO），可改为 OvR |
        | 概率输出 | 设置 `probability=True` |
        | 优缺点 | 强大但慢，适合中小规模数据 |
        | 大规模替代 | LinearSVC、SGDClassifier |

    如果你有具体的使用场景（如文本分类、图像识别、不平衡数据等），我可以进一步给出针对性的调参建议！

* sklearn tutorial, 随机森林的 example

    随机森林的 example:

    ```py
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    def main():
        clf = RandomForestClassifier(random_state=0)
        X = [[1, 2, 3],
             [11, 12, 13]]
        y = [0, 1]
        clf.fit(X, y)
        pred = clf.predict(X)
        print(pred)  # array([0, 1])
        clf.predict([[4, 5, 6], [14, 15, 16]])
        print(pred)  # array([0, 1])
        return

    if __name__ == '__main__':
        main()
    ```

    如果这段代码可以跑通，说明 sklearn 环境没有问题。

    sklearn 中默认 X 的 shape 为`(num_samples, num_features)` 
    
    y 通常是个 vector，行或列无所谓，用行就行。第 i 个元素代表 X 中第 i 个 sample 的 target.

## Topics

### metrics

* accuracy_score() 的用法

    1. 函数作用

        `accuracy_score` 用于计算**分类准确率**，即：

        $$
        \text{准确率} = \frac{\text{预测正确的样本数}}{\text{总样本数}}
        $$

        它返回一个浮点数（或一个数组，取决于参数设置），表示模型预测的准确程度。

    2. 导入方式

        ```py
        from sklearn.metrics import accuracy_score
        ```

    3. 函数签名与参数

        ```py
        accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)
        ```

        **参数说明：**

        | 参数 | 类型 | 说明 |
        |------|------|------|
        | `y_true` | 1d array-like | 真实标签 |
        | `y_pred` | 1d array-like | 预测标签 |
        | `normalize` | bool, 默认 True | 若为 True，返回准确率（0~1 之间的浮点数）；若为 False，返回预测正确的样本数（整数） |
        | `sample_weight` | array-like, 可选 | 每个样本的权重，用于计算加权准确率 |

    4. 基本用法示例

        1. 二分类

            ```py
            from sklearn.metrics import accuracy_score

            y_true = [0, 1, 1, 0, 1]
            y_pred = [0, 1, 0, 0, 1]

            acc = accuracy_score(y_true, y_pred)
            print(acc)          # 0.8 （4/5 正确）
            ```

        2. 多分类

            ```py
            y_true = [0, 1, 2, 2, 1]
            y_pred = [0, 2, 2, 1, 1]

            print(accuracy_score(y_true, y_pred))  # 0.6 （3/5 正确）
            ```

        3. 返回正确样本数（`normalize=False`）

            ```py
            acc_count = accuracy_score(y_true, y_pred, normalize=False)
            print(acc_count)   # 3（正确个数）
            ```

    5. 样本权重（sample_weight）

        当样本重要程度不同时，可以给每个样本赋权。

        ```python
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]  # 第3个样本预测错误

        # 默认权重都是1，准确率为 3/4 = 0.75
        print(accuracy_score(y_true, y_pred))  # 0.75

        # 给第3个样本（索引2）高权重，则错误影响更大
        weights = [1, 1, 10, 1]
        print(accuracy_score(y_true, y_pred, sample_weight=weights))  
        # 加权正确数 = 1+1+0+1=3，总权重=13，准确率=3/13≈0.2308
        ```

    6. 注意事项与常见坑

        1. 标签顺序不重要

            `accuracy_score` 只比较对应位置的 `y_true` 和 `y_pred` 是否相等，不关心类别顺序。

        2. 输入格式要求

            - 可以是列表、NumPy 数组、Pandas Series 等。
            - 两者长度必须相等，否则报错。

        3. 多标签（multi-label）问题不支持

            `accuracy_score` 不适用于多标签分类（每个样本有多个标签）。

            对于多标签，应使用 `jaccard_score` 或 `f1_score`（带 `average` 参数）。

        4. 类别不平衡时的局限

            准确率对类别不平衡非常敏感（例如 90% 负例，全预测负类也能得 90% 准确率）。
            
            这种情况建议搭配混淆矩阵、精确率/召回率或 F1-score 一起使用。

    7. 与其他指标的简单对比

        | 指标 | 适用场景 | 优点 | 缺点 |
        |------|----------|------|------|
        | accuracy_score | 平衡分类问题 | 直观、计算快 | 对不平衡敏感 |
        | precision_score | 关注假阳性代价高（如垃圾邮件检测） | 衡量“预测为正的有多准” | 忽视假阴性 |
        | recall_score | 关注假阴性代价高（如疾病筛查） | 衡量“正例被找出多少” | 忽视假阳性 |
        | f1_score | 需要平衡 P/R | 调和平均数 | 不直观 |

    8. 完整代码总结

        ```python
        from sklearn.metrics import accuracy_score
        import numpy as np

        y_true = np.array([0, 1, 2, 2, 1])
        y_pred = np.array([0, 2, 2, 1, 1])

        # 默认准确率
        print(accuracy_score(y_true, y_pred))          # 0.6

        # 返回正确个数
        print(accuracy_score(y_true, y_pred, normalize=False))  # 3

        # 加权版本
        weights = [1, 1, 0.5, 1, 1]
        print(accuracy_score(y_true, y_pred, sample_weight=weights))  
        # 正确加权和 = 1+1+0.5+0+1 = 3.5，总权重4.5，结果≈0.7778
        ```

* 详细讲讲 `classification_report()` 的用法

    `classification_report()` 是分类任务中**最全面的单函数评估工具**，能一次性输出多个核心指标。

    1. 函数作用

        `classification_report` 用于构建一个**文本报告**，展示每个类别的：

        - **精确率（Precision）**
        - **召回率（Recall）**
        - **F1-score**
        - **支持数（Support）**

        它帮助您快速了解模型在每个类别上的表现，而不仅仅是总体准确率。

    2. 导入方式

        ```python
        from sklearn.metrics import classification_report
        ```

    3. 函数签名与参数

        ```python
        classification_report(
            y_true,
            y_pred,
            *,
            labels=None,
            target_names=None, 
            sample_weight=None,
            digits=2,
            output_dict=False, 
            zero_division='warn'
        )
        ```

        **核心参数详解：**

        * `y_true`: 1d array-like, 真实标签
        * `y_pred`: 1d array-like, 预测标签
        * `labels`: list, 可选, 指定要报告的类别列表（默认使用所有类别）
        * `target_names`, list, 可选, 类别的显示名称（用于美化输出）
        * `sample_weight`, array-like, 可选, 样本权重
        * `digits`: int, 默认 2, 保留小数位数
        * `output_dict`: bool, 默认 False, 若为 True，返回字典而非字符串
        * `zero_division`: str, 默认 'warn', 当分母为0时的处理方式：'warn'（警告并返回0）、'0'（返回0）、'1'（返回1）

    4. 基本用法示例

        1. 二分类

            ```python
            from sklearn.metrics import classification_report

            y_true = [0, 1, 0, 1, 0, 1, 0, 1]
            y_pred = [0, 1, 0, 0, 0, 1, 1, 1]

            print(classification_report(y_true, y_pred))
            ```

            **输出：**

            ```
                        precision    recall  f1-score   support

                    0       0.67      0.75      0.71         4
                    1       0.75      0.67      0.71         4

                accuracy                           0.71         8
            macro avg       0.71      0.71      0.71         8
            weighted avg       0.71      0.71      0.71         8
            ```

        2. 多分类

            ```python
            y_true = [0, 1, 2, 2, 1, 0, 2, 1]
            y_pred = [0, 2, 2, 1, 1, 0, 1, 1]

            print(classification_report(y_true, y_pred))
            ```

            **输出：**

            ```
                        precision    recall  f1-score   support

                    0       1.00      1.00      1.00         2
                    1       0.67      1.00      0.80         3
                    2       0.50      0.33      0.40         3

                accuracy                           0.75         8
            macro avg       0.72      0.78      0.73         8
            weighted avg       0.69      0.75      0.70         8
            ```

    5. 参数详解与高级用法

        1. 使用 `target_names` 自定义类别名称

            当类别是数字时，可以替换为有意义的名称：

            ```python
            y_true = [0, 1, 2, 2, 1]
            y_pred = [0, 2, 2, 1, 1]

            print(classification_report(
                y_true, 
                y_pred, 
                target_names=['苹果', '香蕉', '橙子']
            ))
            ```

            **输出：**

            ```
                        precision    recall  f1-score   support

                    苹果       1.00      1.00      1.00         1
                    香蕉       0.50      1.00      0.67         2
                    橙子       0.50      0.50      0.50         2

                accuracy                           0.60         5
            macro avg       0.67      0.83      0.72         5
            weighted avg       0.60      0.60      0.57         5
            ```

        2. 使用 `labels` 指定报告的类别顺序

            ```python
            # 强制按 [2, 0, 1] 顺序报告
            print(classification_report(
                y_true, 
                y_pred, 
                labels=[2, 0, 1],
                target_names=['橙子', '苹果', '香蕉']
            ))
            ```

        3. 使用 `digits` 控制精度

            ```python
            print(classification_report(y_true, y_pred, digits=4))
            ```

        4. 使用 `output_dict` 返回字典（便于程序处理）

            ```python
            report_dict = classification_report(
                y_true, 
                y_pred, 
                output_dict=True
            )

            print(report_dict['0'])          # 类别0的指标
            print(report_dict['accuracy'])   # 准确率
            print(report_dict['macro avg'])  # 宏平均
            ```

            **返回的字典结构：**

            ```python
            {
                '0': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1},
                '1': {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.666..., 'support': 2},
                '2': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.5, 'support': 2},
                'accuracy': 0.6,
                'macro avg': {'precision': 0.666..., 'recall': 0.833..., 'f1-score': 0.722..., 'support': 5},
                'weighted avg': {'precision': 0.6, 'recall': 0.6, 'f1-score': 0.566..., 'support': 5}
            }
            ```

        5. 使用 `sample_weight` 加权

            ```python
            weights = [1, 1, 0.5, 0.5, 1]  # 第3、4个样本权重减半
            print(classification_report(y_true, y_pred, sample_weight=weights))
            ```

        6. 处理 `zero_division`（防止除以零）

            当某个类别在真实标签或预测标签中完全不存在时：

            ```python
            y_true = [0, 0, 0]
            y_pred = [0, 0, 1]  # 类别1从未真实出现

            # 默认会警告，并给类别1的 precision 设为 0.0
            print(classification_report(y_true, y_pred))

            # 安静模式，返回 0
            print(classification_report(y_true, y_pred, zero_division=0))

            # 或者返回 1
            print(classification_report(y_true, y_pred, zero_division=1))
            ```

    6. 报告中的指标详解

        1. 每类指标

            假设二分类，关注类别 A：

            | 指标 | 公式 | 含义 |
            |------|------|------|
            | **Precision** | TP / (TP + FP) | 预测为 A 的样本中，真正是 A 的比例 |
            | **Recall** | TP / (TP + FN) | 真正的 A 中，被正确预测出来的比例 |
            | **F1-score** | 2 × (P × R) / (P + R) | Precision 和 Recall 的调和平均 |
            | **Support** | TP + FN | 该类在真实标签中的样本数 |

        2. 总体平均指标

            | 指标 | 计算方式 | 适用场景 |
            |------|----------|----------|
            | **accuracy** | (TP+TN) / 总数 | 总体正确率 |
            | **macro avg** | 对每个类别的指标**简单平均** | 每个类别同等重要（不考虑类别不平衡） |
            | **weighted avg** | 按每个类别的 support 加权平均 | 类别不平衡时更合理 |

        **举例说明区别：**

        ```python
        # 极度不平衡数据
        y_true = [0]*90 + [1]*10
        y_pred = [0]*90 + [0]*10  # 全预测为0

        report = classification_report(y_true, y_pred, output_dict=True)
        print(f"类别0的 recall: {report['0']['recall']}")      # 1.0
        print(f"类别1的 recall: {report['1']['recall']}")      # 0.0
        print(f"macro avg recall: {report['macro avg']['recall']}")  # 0.5
        print(f"weighted avg recall: {report['weighted avg']['recall']}")  # 0.9
        print(f"accuracy: {report['accuracy']}")                # 0.9
        ```

    7. 注意事项与常见坑

        1. 输入必须是一维
            
            不支持多标签分类（multi-label），多标签应使用 `multilabel_confusion_matrix` 等。

        2. 类别顺序影响 `macro avg`
            
            `macro avg` 对每个类别同等对待，即使某个类别样本极少。

            如果类别不平衡严重且想体现多数类影响，看 `weighted avg`。

        3. 与 `accuracy_score` 的区别

            | | `accuracy_score` | `classification_report` |
            |---|---|---|
            | 输出 | 单一数值 | 完整报告（多个指标） |
            | 信息量 | 少 | 丰富 |
            | 适用 | 快速评估 | 深入分析 |
            | 不平衡敏感 | 严重 | 通过 macro/weighted 可缓解 |

    8. 完整实战示例

        ```python
        from sklearn.metrics import classification_report
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        # 生成数据
        X, y = make_classification(n_samples=1000, n_classes=3, weights=[0.7, 0.2, 0.1], random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 训练
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # 预测
        y_pred = clf.predict(X_test)

        # 生成报告
        print(classification_report(
            y_test, 
            y_pred, 
            target_names=['多数类', '中等类', '少数类'],
            digits=3
        ))

        # 获取字典用于后续分析
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        print(f"\n少数类的 F1: {report_dict['少数类']['f1-score']:.3f}")
        print(f"加权平均精确率: {report_dict['weighted avg']['precision']:.3f}")
        ```

    9. 总结

        - `classification_report` 是**分类问题分析的标准起点**
        - 同时提供 P/R/F1，比 `accuracy_score` 信息丰富得多
        - 通过 `macro avg` 和 `weighted avg` 可以判断模型在不同类别上的平衡性
        - 使用 `output_dict=True` 可以将结果集成到自动化流程中

    ---

    如果你还想了解：
    
    - 如何处理多标签分类（multi-label）的评估？
    - 如何画出每个类别的混淆矩阵？
    - 如何自定义指标添加到报告中？

    欢迎继续问我，我可以再展开讲解！

### 常用模型

* linear regression

    ```py
    from sklearn import linear_model
    reg = linear_model.LinearRegression()

    X = [[0, 0],
         [1, 1],
         [2, 2]]
    y = [0, 1, 2]
    reg.fit(X, y)
    coef = reg.coef_
    inte = reg.intercept_

    print('coef: {}'.format(coef))
    print('inte: {}'.format(inte))
    ```

    output:

    ```
    coef: [0.5 0.5]
    inte: 1.1102230246251565e-16
    ```

    `X`第一列是$x_1$，第二列是$x_2$。$y = c_0 + c_1 \cdot x_1 + c_2 \cdot x_2$

    拟合得到的结果`coef`即$[c_1, c_2]$，而`inte`即$c_0$

    对于二个 x 变量的线性拟合，得到的是一个平面，使得 y 到平面沿 y 轴方向的距离的平方和最小。

    对于一个 x 变量的线性拟合，得到的是一条直线，使得 y 到直线沿 y 轴方向的距离的平方和最小。即最小二乘法 (Ordinary Least Squares)。(这里的 ordinary 是什么意思？)

### 自定义数据集

* StandardScaler() 的用法

    完整路径：`sklearn.preprocessing.StandardScaler`

    `StandardScaler` 是 scikit-learn 中最常用、最基础的数据预处理工具之一。它的作用是对特征数据进行**标准化**（也称为 Z-score 标准化）。

    1. 核心原理

        **标准化公式**：

        对于每个特征（每一列），`StandardScaler` 会计算：

        - 均值（mean）μ
        - 标准差（standard deviation） σ

        然后对每个样本的该特征值 x 进行转换：
        
        $$
        x' = \frac{x - \mu}{\sigma}
        $$

        **转换后的结果**：
        - 均值变为 0
        - 方差变为 1（即标准差为 1）

        这使得不同量纲、不同数量级的特征具有可比性，且对许多机器学习算法的性能有正面影响。

    ---

    2. 基本用法（代码示例）

        ```python
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        # 原始数据（2个样本，3个特征）
        X = np.array([[1, 2, 3],
                    [4, 5, 6]])

        # 1. 创建 StandardScaler 对象
        scaler = StandardScaler()

        # 2. 拟合数据并转换（一步完成）
        X_scaled = scaler.fit_transform(X)

        print(X_scaled)
        ```

        输出（约等于）：
        ```
        [[-1. -1. -1.]
        [ 1.  1.  1.]]
        ```

        每个特征均值为0，标准差为1。

    ---

    3. 分步操作（fit + transform）

        更规范的做法是分开两步，尤其在训练/测试集上：

        ```python
        # 训练集
        X_train = np.array([[1, 2, 3],
                            [4, 5, 6]])

        # 测试集（新数据）
        X_test = np.array([[7, 8, 9],
                        [10, 11, 12]])

        scaler = StandardScaler()

        # 1. 在训练集上拟合（计算均值和标准差）
        scaler.fit(X_train)

        # 2. 转换训练集
        X_train_scaled = scaler.transform(X_train)

        # 3. 使用同样的参数转换测试集（重要！）
        X_test_scaled = scaler.transform(X_test)

        print("训练集均值:", scaler.mean_)
        print("训练集标准差:", scaler.scale_)
        ```

        **关键点**：测试集必须使用训练集的均值和标准差进行转换，而不是重新 fit，否则会导致数据泄露和评估失真。

    ---

    4. 重要属性和方法

        | 属性/方法 | 说明 |
        |---------|------|
        | `fit(X)` | 计算 X 每个特征的均值（`mean_`）和标准差（`scale_`） |
        | `transform(X)` | 使用已计算的均值和标准差进行标准化 |
        | `fit_transform(X)` | 拟合 + 转换（一步完成） |
        | `inverse_transform(X)` | 将标准化后的数据还原为原始尺度 |
        | `mean_` | 每个特征的均值（numpy 数组） |
        | `scale_` | 每个特征的标准差（numpy 数组） |
        | `var_` | 每个特征的方差（`scale_` 的平方） |
        | `n_samples_seen_` | 拟合时使用的样本数 |

    5. 逆变换（还原数据）

        ```python
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 还原
        X_original = scaler.inverse_transform(X_scaled)

        # X_original 应等于原始 X（可能有微小浮点误差）
        ```

    6. 重要参数

        `StandardScaler` 的构造函数主要有两个参数：

        - **`copy`** (bool, 默认 True)  
        是否在转换时复制数据。如果为 False，可能会直接修改输入数据（谨慎使用）。

        - **`with_mean`** (bool, 默认 True)  
        是否将数据中心化（减去均值）。  
        设为 `False` 时，只除以标准差（相当于缩放，不中心化）。  
        适用于稀疏矩阵（因为减去均值会破坏稀疏性）。

        - **`with_std`** (bool, 默认 True)  
        是否将数据缩放到单位方差（除以标准差）。  
        设为 `False` 时，只减去均值（中心化），不缩放。

        **示例**：只缩放，不中心化（适合稀疏数据）
        ```python
        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X)
        ```

    7. 使用场景与注意事项

        ✅ **什么时候使用 StandardScaler？**

        - 线性模型（如线性回归、逻辑回归、SVM）
        - 基于距离的算法（KNN、K-Means）
        - 梯度下降优化的算法（神经网络、PCA）
        - 任何假设特征服从正态分布或需要统一量纲的场景

        ❌ **什么时候不用（或慎用）？**

        - 树模型（决策树、随机森林、XGBoost）通常不受特征尺度影响，可以不标准化
        - 特征本身具有明确物理意义且不希望改变时

        ⚠️ **注意事项**

        1. **异常值敏感**：均值和标准差受异常值影响大，如果数据有较多异常值，可考虑使用 `RobustScaler`（基于中位数和四分位数）。
        2. **只对数值特征**：类别特征需要其他编码方式。
        3. **训练/测试一致**：永远先 `fit` 训练集，再用同一个 scaler 去 `transform` 测试集。

    8. 与管道（Pipeline）结合使用

        在实际项目中，通常将 `StandardScaler` 放在 `Pipeline` 中，避免数据泄露：

        ```python
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC())
        ])

        pipe.fit(X_train, y_train)
        pipe.predict(X_test)
        ```

        这样在交叉验证时，每次训练折都会重新拟合 scaler，测试折使用该折的统计量，更严谨。

    9. 完整实战示例

        ```python
        import pandas as pd
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score

        # 加载数据
        data = load_iris()
        X, y = data.data, data.target

        # 划分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 建模
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_scaled, y_train)

        # 预测与评估
        y_pred = knn.predict(X_test_scaled)
        print("准确率:", accuracy_score(y_test, y_pred))
        ```

    10. 总结

        | 要点 | 描述 |
        |------|------|
        | 作用 | 将每个特征转换为均值0，方差1 |
        | 公式 | (x - μ) / σ |
        | 核心方法 | `fit()`, `transform()`, `fit_transform()` |
        | 关键属性 | `mean_`, `scale_` |
        | 逆变换 | `inverse_transform()` |
        | 适用算法 | 线性模型、距离模型、神经网络等 |
        | 注意 | 只 fit 训练集，测试集用同一个 scaler 的 transform |

    如果你还有具体的使用场景（比如大数据集、稀疏数据、在线学习等），我可以进一步展开。

* pipeline

    ```py
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # create a pipeline object
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )

    # load the iris dataset and split it into train and test sets
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # fit the whole pipeline
    pipe.fit(X_train, y_train)

    # we can now use it like any other estimator
    score = accuracy_score(pipe.predict(X_test), y_test)
    print(score)  # 0.9736842105263158
    ```

* sklearn 中的 transormers and pre-processors

    ```py
    from sklearn.preprocessing import StandardScaler
    X = [[0, 15],
         [1, -10]]
    # scale data according to computed scaling values
    out = StandardScaler().fit(X).transform(X)
    print(out)
    # array([[-1.,  1.],
    #        [ 1., -1.]])
    ```

    注：

    1. transforms 竟然都继承于 BaseEstimator，这样确实方便了链式调用，但是类的意义变不对了

* `train_test_split()`

    它是 scikit-learn 中最常用、最基础的数据划分工具。

    * 函数原型

        ```py
        sklearn.model_selection.train_test_split(
            *arrays,
            test_size=None,
            train_size=None,
            random_state=None,
            shuffle=True,
            stratify=None
        )
        ```

    * 参数详解

        * `*arrays`（位置参数）

            * 接受一个或多个数组/矩阵（如特征矩阵 `X` 和标签向量 `y`）。

            * 它们必须具有相同的**第一维度（样本数）**。

            * 常见用法：`train_test_split(X, y)` 或 `train_test_split(X, y, group_ids)`。

        * `test_size`（浮点数或整数，默认 `None`）：
            
            - 决定测试集的大小。
            - 如果为浮点数（0.0~1.0），表示测试集占总样本的比例。
            - 如果为整数，表示测试集的绝对样本数量。
            - 如果为 `None`，则使用 `train_size` 的补集（若 `train_size` 也为 `None`，则默认 `test_size=0.25`）。

        * `train_size`（浮点数或整数，默认 `None`）：

            - 决定训练集的大小，规则同 `test_size`。
            - 通常只设置 `test_size`，让 `train_size` 自动为 `1 - test_size`。

        * `random_state`（整数或 `None`，默认 `None`）：

            - 控制随机打乱过程的随机种子。

            - 设置固定整数（如 `42`）可保证每次运行得到相同的划分结果，便于复现实验。

        * `shuffle`（布尔值，默认 `True`）：

            - 是否在划分前对数据进行随机打乱。

            - 如果数据本身是时间序列或有序的，通常设为 `False`。

            - 若 `stratify` 有值，则 `shuffle` 必须为 `True`（内部强制）。

        * `stratify`（数组或 `None`，默认 `None`）：

            - 用于**分层抽样**的标签数组。

            - 传入 `y` 后，函数会保证训练集和测试集中各类别（分类问题）的比例与原始数据集一致。

            - 对不平衡数据集非常关键。

    * 返回值

        返回一个**元组（tuple）**，顺序为：

        ```py
        X_train, X_test, y_train, y_test
        ```

        或更一般地：

        ```py
        array1_train, array1_test, array2_train, array2_test, ...
        ```

        每个返回的数组都是原数组的**随机子集**，且训练集和测试集互不重叠。

    * 典型用法（代码示例）

        * 示例 1：基本用法（二分类）

            ```python
            from sklearn.model_selection import train_test_split
            from sklearn.datasets import load_iris
            from sklearn.svm import SVC

            # 加载数据
            X, y = load_iris(return_X_y=True)

            # 划分（默认 75% 训练，25% 测试）
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            print(f"训练集样本数: {X_train.shape[0]}")
            print(f"测试集样本数: {X_test.shape[0]}")

            # 训练并评估
            clf = SVC().fit(X_train, y_train)
            print(f"测试集准确率: {clf.score(X_test, y_test):.3f}")
            ```

        * 示例 2：分层抽样（处理不平衡数据）

            ```python
            from sklearn.datasets import make_classification

            # 生成一个不平衡数据集（类别 0: 900, 类别 1: 100）
            X, y = make_classification(
                n_samples=1000, weights=[0.9, 0.1], random_state=42
            )

            # 普通划分（可能使测试集中类别 1 更少）
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print("普通划分 - 训练集类别1比例:", y_train.mean())
            print("普通划分 - 测试集类别1比例:", y_test.mean())

            # 分层划分（保持比例）
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print("分层划分 - 训练集类别1比例:", y_train.mean())
            print("分层划分 - 测试集类别1比例:", y_test.mean())
            ```

            输出会显示分层划分后训练/测试集的类别比例几乎一致（约 0.1）。

        * 示例 3：同时划分多个数组（如特征、标签、样本权重）

            ```python
            import numpy as np

            X = np.random.rand(100, 5)
            y = np.random.randint(0, 2, 100)
            sample_weight = np.random.rand(100)

            X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
                X, y, sample_weight, test_size=0.2, random_state=7
            )

            print(len(X_train), len(y_train), len(sw_train))  # 均为 80
            ```

        * 示例 4：时间序列数据（不随机打乱）

            ```py
            # 假设数据是按时间排序的
            time_series_X = np.arange(1000).reshape(-1, 1)
            time_series_y = np.arange(1000)

            # 取前 80% 作为训练，后 20% 作为测试
            X_train, X_test, y_train, y_test = train_test_split(
                time_series_X, time_series_y, 
                test_size=0.2, shuffle=False, random_state=None
            )

            print("训练集最后一条:", X_train[-1], "测试集第一条:", X_test[0])
            # 输出: 训练集最后一条: [799] 测试集第一条: [800]
            # 保持顺序不交叉
            ```

    * 注意事项与常见陷阱

        * 数据泄露
        
            不要在划分前做任何基于全数据的预处理（如标准化），应先 `fit` 在训练集上，再 `transform` 测试集

        * 小样本数据
        
            如果样本极少，测试集会过小导致评估方差大，建议使用交叉验证。
            
        * 分类问题的分层
        
            若 `y` 是多分类，`stratify=y` 同样有效，但必须保证每个类别至少有一个样本。
        
        * 回归问题
        
            不能使用 `stratify`（除非手动将连续值离散化），可采用 `sklearn.model_selection.StratifiedShuffleSplit` 或自定义分箱。

        * 随机种子
        
            调参时固定 `random_state`，最终评估时可换多个种子取平均

            注：
            
            1. `train_test_split()` 即使不指定 `random_state`，也会每次都使用不同的随机状态种子

### 内置数据集

* iris

    sklearn 中的 iris 数据集是个 toy 数据集，即不需要联网下载，内置的数据集

    * 简介：

        Classes: 3

        Samples per class: 50

        Samples total: 150

        Dimensionality (num features): 4

        Features: real, positive
    
    * `load_iris()`函数的普通模式，会返回一个类字典的数据结构（Bunch）

        这个数据结构不只有字符串形式的 key，还直接有成员名形式的字段。有点像 matlab 中的 struct

        example:

        ```py
        from pandas._libs.lib import is_range_indexer
        import sklearn.datasets as D

        iris_data = D.load_iris()

        keys = iris_data.keys()
        vals = iris_data.values()
        print('keys: {}'.format(keys))

        print()
        print('--------')
        print()
        print('data:')
        data = iris_data['data']
        print(type(data))
        print(data.shape)
        print()

        print('target:')
        target = iris_data.target
        print(type(target))
        print(target.shape)
        print()

        print('feature_names:')
        feature_names = iris_data['feature_names']
        print(type(feature_names))
        print('len: {}'.format((len(feature_names))))
        print(feature_names)
        print()

        print('target_names:')
        target_names = iris_data.target_names
        print(type(target_names))
        print(target_names.shape)
        print(target_names)
        print()

        print('frame:')
        frame = iris_data.frame
        print(frame)
        print()

        print('DESCR:')
        DESCR = iris_data.DESCR
        print('{}'.format(DESCR))
        print()

        print('filename:')
        filename = iris_data.filename
        print('filename: {}'.format(filename))
        print()

        print('End.')
        ```

        output:

        ```
        keys: dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

        --------

        data:
        <class 'numpy.ndarray'>
        (150, 4)

        target:
        <class 'numpy.ndarray'>
        (150,)

        feature_names:
        <class 'list'>
        len: 4
        ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

        target_names:
        <class 'numpy.ndarray'>
        (3,)
        ['setosa' 'versicolor' 'virginica']

        frame:
        None

        DESCR:
        .. _iris_dataset:

        Iris plants dataset
        --------------------

        **Data Set Characteristics:**

        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica

        :Summary Statistics:

        ============== ==== ==== ======= ===== ====================
                        Min  Max   Mean    SD   Class Correlation
        ============== ==== ==== ======= ===== ====================
        sepal length:   4.3  7.9   5.84   0.83    0.7826
        sepal width:    2.0  4.4   3.05   0.43   -0.4194
        petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
        petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
        ============== ==== ==== ======= ===== ====================

        :Missing Attribute Values: None
        :Class Distribution: 33.3% for each of 3 classes.
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988

        The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
        from Fisher's paper. Note that it's the same as in R, but not as in the UCI
        Machine Learning Repository, which has two wrong data points.

        This is perhaps the best known database to be found in the
        pattern recognition literature.  Fisher's paper is a classic in the field and
        is referenced frequently to this day.  (See Duda & Hart, for example.)  The
        data set contains 3 classes of 50 instances each, where each class refers to a
        type of iris plant.  One class is linearly separable from the other 2; the
        latter are NOT linearly separable from each other.

        .. dropdown:: References

          - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
            Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
            Mathematical Statistics" (John Wiley, NY, 1950).
          - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
            (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
          - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
            Structure and Classification Rule for Recognition in Partially Exposed
            Environments".  IEEE Transactions on Pattern Analysis and Machine
            Intelligence, Vol. PAMI-2, No. 1, 67-71.
          - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
            on Information Theory, May 1972, 431-433.
          - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
            conceptual clustering system finds 3 classes in the data.
          - Many, many more ...


        filename:
        filename: iris.csv

        End.
        ```

    * `load_iris()`函数的 `return_X_y` 模式

        直接返回用于训练的 feature 和 label
        
        * feature (X) shape: `(150, 4)`
        
        * label (y) shape: `(150, )`

        ```py
        import sklearn.datasets as D

        iris_data = D.load_iris(return_X_y=True)

        print(type(iris_data))

        X, y = iris_data
        print('X type: {}, shape: {}'.format(type(X), X.shape))
        print('y type: {}, shape: {}'.format(type(y), y.shape))
        ```

        output:

        ```
        <class 'tuple'>

        Press ENTER or type command to continue
        <class 'tuple'>
        X type: <class 'numpy.ndarray'>, shape: (150, 4)
        y type: <class 'numpy.ndarray'>, shape: (150,)
        ```

* toy dataset list

    其他的 toy dataset list: <https://scikit-learn.org/stable/datasets/toy_dataset.html#toy-datasets>

    除了 toy dataset 之外的 dataset: <https://scikit-learn.org/stable/datasets.html>

### 常用函数


## note

安装：

`pip install -U scikit-learn`

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[1, 2, 3],
    [11, 12, 13]]  # 2 samples, 3 features
y = [0, 1]  # classes of each sample
clf.fit(X, y)
clf.predict(X)  # predict classes of the training data
clf.predict([[4, 5, 6], [14, 15, 16]])  # predict classes of new data
```

transformers and pre-processors:

这些模块可以用 pipeline 预处理数据。和 mmdet 挺像的，不知道 sklearn 和 mmdet 他俩谁抄的谁。

```python
from sklearn.preprocessing import StandardScaler
X = [[0, 15],
    [1, -10]]
# scale data according to computed scaling values
StandardScaler().fit(X).transform(X)
```

输出：

```
array([-1., 1.],
    [1., -1.])
```

注：

1. pre-processors, transformers 和 extimators 都继承自`BaseEstimator`。感觉这个设计挺智障的。

pipelines:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticREgression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create a pipeline object
pipe = make_pipeline(
    StandardScaler(),
    LOgisticRergression()
)

# Load the iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# fit the whole pipeline
pipe.fit(X_train, y_train)
accuracy_score(pipe.predict(X_test), y_test)
```

Model evaluation:

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X, y = make_regression(n_samples=1000, random_state=0)
lr = LinearRegression()

result = cross_validate(lr, X, y)  # defaults to 5-fold CV
result['test_score']  # r_squared score is high because dataset is easy
```

Automatic parameter searches

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizeSearchCV, train_test_split
from scipy.stats import randint

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# define the parameter space that will be searched over
param_distributions = {'n_estimators': randint(1, 5), 'max_depth': randint(5, 10)}

# now create a searchCV object and fit it to the data
search = RandomizedSearchCV(estimator=RandomForestRegression(random_state=0), n_iter=5, param_distributions=param_distributions, random_state=0)
search.fit(X_train, y_train)
search.best_params_

# the search object now acts like a normal random forest estimator
# with max_depth=9 and n_estimators=4
search.score(X_test, y_test)
```
