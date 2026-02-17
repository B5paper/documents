# Machine Learning Note

## cache

* 简述什么是 AI 中的对比学习

    **对比学习（Contrastive Learning）**是人工智能领域，尤其是自监督学习中的一种重要方法，其核心思想是通过**学习区分相似（正）样本与不相似（负）样本**来获得高质量的数据表示。

    **核心思想**

    1. **无需人工标注**：通过自动构造正负样本对，从无标签数据中学习。

    2. **核心原则**：在表示空间中，**拉近相似样本**（正样本对），**推远不相关样本**（负样本对）。

    **基本流程**

    1. **数据增强**：对同一原始数据（如图像）进行两次随机变换（如裁剪、旋转、颜色调整），生成两个相似视图作为**正样本对**。

    2. **构建负样本**：将不同原始数据生成的样本视为**负样本对**。

    3. **编码与对比**：

        * 通过编码器（如神经网络）提取样本的特征表示。

        * 使用**对比损失函数**（如InfoNCE Loss）优化模型，使正样本对表示相似度最大化，负样本对相似度最小化。

    **关键组件**

    * **正样本对**：同一数据的不同增强版本。
    * **负样本对**：不同数据的增强版本。
    * **损失函数**：常用**InfoNCE Loss**，公式鼓励正样本相似度远高于负样本。

    **经典方法**

    * **SimCLR**：简化框架，强调数据增强和投影头的重要性。
    * **MoCo**：引入动量编码和动态字典，高效构建大量负样本。
    * **BYOL**：无需负样本，通过不对称网络结构避免模型坍塌。

    **优势**

    * **无需标注数据**，利用海量无标签数据学习通用表示。
    * **学习到的表示可迁移**，在下游任务（如分类、检测）上微调即可取得优异性能。
    * **对数据增强鲁棒**，能捕捉数据中的本质特征。

    **应用领域**

    * **计算机视觉**：图像分类、目标检测、语义分割。
    * **自然语言处理**：文本表示学习（如SimCSE）。
    * **多模态学习**：CLIP模型通过对比学习对齐图像和文本表示。

    **总结**

    对比学习通过**实例区分任务**驱动模型学习数据的内在结构和不变特征，已成为自监督学习的代表性范式，显著提升了模型在无标注或少量标注数据下的表现能力。其核心在于**通过对比构建监督信号，让模型学会“什么更像，什么不像”**。

* 对比学习有哪些经典论文与项目

    以下是AI对比学习领域具有里程碑意义的**经典论文**和**开源项目**，它们推动了该方向的发展。

    **一、经典论文**

    1. 奠基与理论

        - **《A Simple Framework for Contrastive Learning of Visual Representations》（SimCLR）**  

            **作者**：Ting Chen等（Google Research，2020）  

            **贡献**：提出简单有效的框架，强调**数据增强组合**与**非线性投影头**的关键作用。  

            **核心**：端到端学习，大批量负样本。

        - **《Momentum Contrast for Unsupervised Visual Representation Learning》（MoCo）**  

            **作者**：Kaiming He等（FAIR，2019）  

            **贡献**：引入**动量编码器**和**动态队列**，构建大规模且一致的负样本库，解决大批量训练需求。

        - **《Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning》（BYOL）**  

            **作者**：DeepMind（2020）  

            **贡献**：**无需负样本**，通过不对称网络结构和动量更新避免模型坍塌，挑战了对比学习必须依赖负样本的认知。

    2. 改进与拓展

        - **《Improved Baselines with Momentum Contrastive Learning》（MoCo v2）**  

            **作者**：FAIR（2020）  

            **贡献**：融合SimCLR的数据增强和投影头设计，显著提升MoCo性能。

    - **《Exploring Simple Siamese Representation Learning》（SimSiam）**  

        **作者**：Kaiming He等（2020）  

        **贡献**：极简孪生网络，无需负样本、动量编码器或大批量，仅靠**停止梯度**避免坍塌。

    - **《Contrastive Language-Image Pre-training》（CLIP）**  

        **作者**：OpenAI（2021）  

        **贡献**：**多模态对比学习**，将图像和文本编码到共享空间，实现零样本迁移。

    3. 自然语言处理应用

        - **《SimCSE: Simple Contrastive Learning of Sentence Embeddings》**  

            **作者**：Danqi Chen等（Princeton，2021）  

            **贡献**：通过**Dropout噪声**构建正样本，提升句子表示的语义一致性。

        - **《Contrastive Learning for Neural Machine Translation》**  

            **应用**：在机器翻译中引入对比损失，提升翻译质量与鲁棒性。

    **二、重要开源项目**

    1. 框架与代码库

        - **PyTorch Lightning Bolts**  

            **链接**：https://github.com/Lightning-AI/lightning-bolts  

            **内容**：包含SimCLR、MoCo、BYOL等预实现模型，易于上手。

        - **Facebook Research - MoCo**  

            **链接**：https://github.com/facebookresearch/moco  

            **内容**：官方MoCo系列实现（v1/v2/v3）。

        - **SimCLR（Google Research）**  

            **链接**：https://github.com/google-research/simclr  

            **内容**：官方TensorFlow实现，含训练脚本。

        - **OpenCLIP**  

            **链接**：https://github.com/mlfoundations/open_clip  

            **内容**：CLIP的开源复现与扩展，支持多种预训练模型。

    2. 工具包与基准

        - **VISSL（FAIR）**  

            **链接**：https://github.com/facebookresearch/vissl  

            **内容**：包含多种自监督学习算法的PyTorch库，支持快速实验。

        - **Contrastive Learning Survey Papers**  

            **链接**：https://github.com/HobbitLong/CMC  

            **内容**：经典论文列表与资源汇总（如CMC、InfoMin等）。

    三、关键进展时间线

    | 年份 | 模型      | 核心创新点                           |
    |------|-----------|--------------------------------------|
    | 2018 | CPC       | 基于预测的对比学习（语音/图像）      |
    | 2019 | MoCo      | 动量对比、动态负样本队列             |
    | 2020 | SimCLR    | 数据增强组合 + 投影头                |
    | 2020 | BYOL      | 无需负样本，不对称网络               |
    | 2020 | SwAV      | 在线聚类替代负样本                   |
    | 2021 | CLIP      | 多模态对比学习（图文对齐）           |
    | 2021 | SimSiam   | 极简孪生网络，仅需停止梯度           |

    四、学习建议

    1. **入门**：从**SimCLR**或**MoCo**论文入手，理解基础框架。

    2. **深入**：阅读**BYOL**和**SimSiam**，思考如何避免坍塌。

    3. **应用**：尝试**CLIP**或**SimCSE**，了解跨模态/文本对比学习。

    4. **实践**：使用开源项目（如VISSL）在标准数据集（CIFAR-10/ImageNet）上复现实验。

    这些工作共同构建了对比学习的理论基础与实践框架，推动了自监督学习的广泛应用。

* 对比学习存在的问题或缺点是什么

    对比学习虽然强大，但并非完美。以下是其存在的主要**问题与缺点**，既有方法论上的局限，也有实际应用中的挑战：

    一、核心方法论缺陷

    1. 对负样本的强依赖（大部分早期方法）

        *   **问题**：经典的对比损失（如InfoNCE）需要大量**负样本**来防止模型坍塌（即所有输入映射到同一个点）。获取足够多且高质量的负样本是性能关键。

        *   **后果**：

            *   **计算与存储成本高昂**：需要维护大批量的负样本或庞大的队列/字典，对GPU内存和计算力要求极高。

            *   **“假阴性”问题**：随机采样的负样本中，可能包含与锚点语义相似的样本（例如，同一物体的不同图片、同义词句子）。模型被强制推开这些本应靠近的样本，**损害了表示质量**。

            *   **对批量大小敏感**：性能通常随批量大小增加而提升，但超大批量在现实中难以实现。

    2. 数据增强的“双刃剑”效应

        *   **问题**：对比学习的正样本对高度依赖于**预设的数据增强策略**。

        *   **后果**：

            *   **领域敏感**：在图像领域有效的增强（如裁剪、颜色抖动）未必适用于其他领域（如医疗影像、文本、音频）。需要大量领域知识来设计有效的增强方法。

            *   **可能破坏语义**：过于激进的增强可能改变原始数据的语义（例如，将猫的图片裁剪得只剩尾巴），导致模型学习到无关或不稳定的特征。

    3. 信息利用效率偏低

        *   **问题**：对比学习本质是**二元判别任务**（相似/不相似），忽略了数据内部更丰富的结构和层次关系。

        *   **后果**：模型可能只学习到区分类别的“最显著特征”，而忽略了更精细的、具有判别力的子类别特征或属性。

    4. 评价指标与最终任务的潜在鸿沟

        *   **问题**：预训练阶段的优化目标是**对比损失**，但下游任务（如分类、检测）的目标不同。

        *   **后果**：在对比学习任务上表现好的表示，不一定在特定下游任务上最优，需要进行微调。这揭示了预训练目标与最终应用目标的不完全一致性。

    二、技术实现与效率问题

    1. 训练不稳定与收敛慢

        *   **问题**：对比学习目标函数可能存在**局部最优**或**平坦区域**，导致训练过程不稳定，需要仔细调整超参数（如温度系数 τ、学习率）。

        *   **后果**：调参成本高，复现论文结果难度大。

    2. 高计算成本

        *   **问题**：需要进行大量的成对相似度计算（特别是负样本很多时）。

        *   **后果**：从数据加载、增强到前向/反向传播，整个过程比有监督学习更耗时耗能。

    三、后续研究的改进方向（对应上述问题）

    | 问题 | 代表性改进思路 | 例子 |
    | :--- | :--- | :--- |
    | **依赖负样本** | **减少或消除负样本** | BYOL, SimSiam （通过架构设计避免坍塌） |
    | | **更智能的负样本挖掘** | 硬负样本挖掘， 去噪对比学习 |
    | **“假阴性”问题** | **引入更细粒度的相似度** | 使用连续相似度而非二元标签， 引入知识蒸馏 |
    | | **聚类或原型对比** | SwAV, ProtoNCE （在聚类中心层面进行对比） |
    | **数据增强依赖** | **自动学习增强策略** | AutoAugment 用于对比学习 |
    | | **多模态/跨模态作为自然增强** | CLIP （图文互为正样本） |
    | **信息利用不足** | **引入聚类或冗余减少** | Barlow Twins （减少特征维度间的冗余） |
    | **计算成本高** | **更高效的负样本管理** | MoCo 的动量队列 |
    | | **使用更小的投影头** | 简化网络设计 |

    四、总结

    对比学习的主要缺点可归纳为：**1) 对负样本的强依赖及其引发的“假阴性”和成本问题；2) 对数据增强策略的高度敏感；3) 信息利用的二元性导致的不够精细化；4) 高昂的计算与优化成本。**

    这些缺点推动了领域的发展，催生了**无需负样本**、**基于聚类**、**跨模态自然监督**以及**更高效架构**等一系列新方法。理解这些局限，有助于在实际任务中更理性地选择、改进或超越对比学习范式。

* 所以对比学习不如作为有监督训练的一个子任务，比如数据增广。单独搞对比学习意义不大

    您的这个观点非常犀利，也代表了AI社区中一种有影响力的声音，尤其是在“对比学习热潮”之后。这个观点可以总结为：**“对比学习的核心贡献——通过数据增强构建自监督信号——与其作为一个独立的预训练范式，不如作为传统监督学习的强力补充（一种高级的数据增强/正则化技术）。”**

    **这个观点有一定道理，但它可能低估了对比学习作为一种“表示学习范式”的独立意义。** 下面我们来客观分析一下两种视角：

    视角一：支持您的观点 —— “对比学习作为高级数据增强/子任务”

    1.  **监督学习仍是主流**：在许多拥有**充足、高质量标注数据**的场景下（例如ImageNet分类），从头开始的监督训练通常能达到最佳性能。对比学习预训练+微调的模式，在计算效率和最终精度上未必能显著超越精心设计的监督模型。
    2.  **最终目标导向**：工业界的核心是解决**具体任务**。如果对比学习预训练不能稳定、显著地提升最终任务指标，那么将其作为一个复杂的、独立的预训练阶段，其**投入产出比**可能确实不如直接使用监督学习，并结合传统的数据增强、正则化、预训练权重（如在ImageNet上监督预训练的模型）等方法。
    3.  **工程复杂度**：对比学习训练本身需要调参、管理负样本、设计增强策略等，引入额外的工程复杂度和不确定性。
    4.  **融合趋势**：确实有研究将对比损失作为**辅助损失**，与主监督损失结合，起到正则化、提升模型鲁棒性和表示质量的作用。这证明了其作为“子任务”或“高级正则化”的有效性。

    **因此，在资源有限、任务明确、且有标注数据的情况下，这个务实观点是非常正确的。**

    视角二：反驳该观点 —— “对比学习作为独立范式意义重大”

    然而，认为“单独搞对比学习意义不大”可能过于绝对，因为它忽视了对比学习解决的**根本性科学问题**及其带来的**范式转移**：

    1.  **解决“标注数据依赖”的核心挑战**：AI发展的最大瓶颈之一就是**高质量标注数据稀缺且昂贵**。对比学习的根本意义在于，它提供了一套系统性的方法论，让我们能**从海量无标注数据中自动学习通用的、高质量的表示**。这是监督学习无法做到的。它的目标不是替代监督学习，而是**解决监督学习的前提依赖问题**。
    2.  **通向通用人工智能的基石**：人类学习很大程度上是基于对世界的对比观察（区分相似与不同），而非依赖大量外部标注。对比学习更贴近这种**自监督、从观察中学习**的范式，是构建更通用、更自主AI系统的重要路径探索。
    3.  **催生了革命性应用**：最有力的反驳例子就是 **CLIP** 和 **DALL-E** 系列。
        *   **CLIP**：通过海量图文对的**跨模态对比学习**，学会了图像和文本的联合表示，实现了强大的**零样本图像分类和检索能力**。这种能力是传统“监督学习+数据增强”完全无法实现的，因为它本质上学会了视觉概念的语义抽象。
        *   **多模态大模型基础**：对比学习是训练像CLIP这样的视觉编码器的关键，而CLIP编码器是现代文生图模型（如Stable Diffusion、DALL-E）和大型多模态模型（如GPT-4V）的**核心组件之一**。其贡献不可替代。
    4.  **提供了新的模型评估视角**：对比学习催生了基于**线性探测**或**少样本探测**的模型评估方法，这能更纯粹地评估模型**特征表示的质量**，而不受微调阶段引入的新参数干扰。
    5.  **推动了表示学习理论发展**：对比学习迫使研究者深入思考“什么构成了好的表示？”、“如何避免模型坍塌？”等基础问题，推动了自监督学习理论的发展（如信息最大化、不变性原理等）。

    结论：并非“意义不大”，而是“意义不同”

    我们可以做一个更精确的总结：

    *   **对于具体任务工程师**：如果您的目标是在一个**有标注数据、定义清晰的任务**上获得最佳性能，那么将对比学习的思想（如精心设计的数据增强、使用对比损失作为正则项）融入到您的监督学习管道中，可能是**最高效、最务实**的选择。从这个角度看，将其视为“高级工具包”而非独立范式，是合理的工程决策。
    *   **对于研究者与通用AI探索者**：对比学习作为一种**独立的表示学习范式**，其意义极为重大。它为解决“无标注数据利用”这一AI核心难题提供了关键思路，并且已经催生了像CLIP这样具有颠覆性能力的模型。它不仅仅是一种数据增强，更是一种**新的学习范式**。

    因此，更准确的表述可能是：

    **对比学习既是监督学习一个强有力的补充工具（可作为高级数据增强/正则化技术），更是一个具有独立价值的、解决“如何从无标注数据中学习”这一根本问题的自监督学习范式。它的终极意义不在于在特定任务上超越监督学习，而在于打开了一条不依赖于海量标注数据就能学习通用知识的新道路。** 忽视其后者，就等于忽视了AI研究的前沿探索方向。

* 表征学习

    表征学习是机器学习的一个分支，其核心目标是自动发现数据中有效的特征表示（Representation），使原始数据转换为更容易被机器学习模型（如分类器、预测模型等）理解和使用的新形式。

    核心思想

    原始数据（如图像像素、文本字符、音频波形）通常具有高维度、稀疏、冗余或难以直接建模的特点。表征学习通过模型自动学习将这些原始数据转化为低维度、稠密、有意义的向量或编码（即“表征”），这些表征应当：

    * 保留重要信息（如语义、结构）。

    * 剔除噪声和冗余。

    * 有利于下游任务（如分类、检测、生成）。

    主要方法与类型

    * 监督表征学习

        * 通过有标签数据的监督任务（如图像分类）训练模型，其隐含层的输出可作为表征。

        * 例如：用预训练的ResNet提取图像特征向量。

    * 无监督表征学习

        * 仅利用无标签数据学习数据的内在结构。

        * 典型方法：

            * 自编码器（Autoencoder）：通过编码-解码重构学习紧凑表征。

            * 对比学习（Contrastive Learning）：如SimCLR、MoCo，通过拉近相似样本、推开不相似样本来学习表征。

            * 生成模型：如VAE、GAN，通过生成过程间接学习数据分布的表征。

    * 自监督学习（Self-supervised Learning）

        * 无监督学习的一种，通过设计“代理任务”自动生成标签来学习表征。

        * 例如：

            * 掩码语言模型（MLM）：如BERT，通过预测被遮盖的词语学习语言表征。

            * 图像拼图：通过预测图像块的相对位置学习视觉表征。

    * 迁移学习与预训练模型

        * 在大规模数据上预训练模型（如ImageNet、Wikipedia），将其学到的通用表征迁移到下游任务，通常只需微调。

    关键优势

    * 减少特征工程：自动学习特征，降低对人工设计特征的依赖。

    * 可迁移性：预训练的表征可泛化到多种任务。

    * 层次化抽象：深度模型能逐层学习从低级到高级的特征（如边缘→纹理→物体部件）。

    应用领域

    * 计算机视觉：图像特征提取、目标检测、人脸识别。

    * 自然语言处理：词向量（Word2Vec）、句向量（BERT、GPT）。

    * 语音处理：声学特征编码。

    * 推荐系统：用户与物品的嵌入表示。

    * 多模态学习：跨文本、图像、音频的统一表征。

    总结

    表征学习是将原始数据转化为“机器更懂”的形式的过程，是现代深度学习的基础。它通过数据驱动的方式自动发现内在规律，是实现通用人工智能的关键技术之一。随着大模型和多模态发展，学习通用、可解释、鲁棒的表征仍是核心挑战。

* 表征学习相关的论文和项目

    一、里程碑与经典论文

    1. 无监督/自监督学习（基石）

        * Word2Vec (2013) - Mikolov et al.

            * 标题：Efficient Estimation of Word Representations in Vector Space

            * 意义：开创性的词向量工作，提出CBOW和Skip-gram模型，表明可以从无标签文本中学习到语义丰富的词表征。

        * Autoencoder (2006重现热潮) - Hinton & Salakhutdinov

            * 标题：Reducing the Dimensionality of Data with Neural Networks

            * 意义：展示了深度自编码器能学习比PCA更好的数据低维表征，推动了深度表征学习。

        * BERT (2018) - Devlin et al.
 
            * 标题：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

            * 意义：基于掩码语言模型的自监督预训练范式，彻底改变了NLP，证明了从纯文本中学习深度上下文表征的强大能力。

        * MoCo (2020) - He et al.

            * 标题：Momentum Contrast for Unsupervised Visual Representation Learning

            * 意义：视觉对比学习的经典之作，提出动量编码器和动态字典，使无监督视觉表征学习接近有监督性能。

        * SimCLR (2020) - Chen et al.
 
            * 标题：A Simple Framework for Contrastive Learning of Visual Representations

            * 意义：简化了视觉对比学习框架，强调了数据增强和投影头的重要性，影响深远。

    2. 理论分析与理解

        * InfoMax Principle (1985) - Linsker

            * 标题：Self-Organization in a Perceptual Network

            * 意义：提出最大化输入与输出之间互信息的原则，是很多自监督学习（如对比学习）的理论基础。

        + “On the Mutual Information Perspective...” (2020) - Tschannen et al.

            * 标题：On Mutual Information Maximization for Representation Learning

            * 意义：批判性地讨论了互信息最大化在实践中的挑战，推动了更务实的理解。

    二、前沿研究方向与论文

    1. 多模态表征学习

        * CLIP (2021) - Radford et al.

            * 标题：Learning Transferable Visual Models From Natural Language Supervision

            * 意义：通过图文对比学习，学习对齐的图像和文本表征，实现零样本分类，开启多模态研究新范式。

        * DALL-E / Stable Diffusion (2021-2022) - Ramesh et al. / Rombach et al.

            * 意义：基于扩散模型的文生图系统，其核心是学习一个能将文本和图像对齐到同一隐空间的强大多模态表征。

    2. 自监督学习新范式

        * MAE (2021) - He et al.

            * 标题：Masked Autoencoders Are Scalable Vision Learners

            * 意义：将BERT的掩码重建思想成功应用于计算机视觉，使用非对称编码器-解码器架构高效学习视觉表征。

        * BYOL (2020) - Grill et al.

            * 标题：Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning

            * 意义：无需负样本的对比学习，仅靠在线网络和目标网络的相互预测进行学习，挑战了对比学习的固有认知。

    3. 图表征学习

        * Node2Vec (2016) - Grover & Leskovec

            * 标题：node2vec: Scalable Feature Learning for Networks

            * 意义：将Word2Vec思想应用于图节点，通过有偏随机游走生成节点序列进行学习。

        * GraphSAGE (2017) - Hamilton et al.

            * 标题：Inductive Representation Learning on Large Graphs

            * 意义：提出一种归纳式框架，通过采样和聚合邻居特征来生成未见节点的表征。

        * Graph Contrastive Learning (2020+) - 如 GRACE, DGI

            * 意义：将对比学习思想应用于图数据，通过破坏图结构（如删边、加噪）来构建正负样本，学习鲁棒图表征。

    三、重要开源项目与工具库

    1. 综合项目/预训练模型库

        * Hugging Face Transformers

            * 链接: https://huggingface.co/docs/transformers

            * 介绍：最流行的NLP库，提供了数千个预训练的语言表征模型（BERT, GPT, T5等）及其微调、使用接口。是多模态和视觉模型也在迅速扩展。

        * PyTorch Image Models (timm)

            * 链接: https://github.com/rwightman/pytorch-image-models

            * 介绍：由Ross Wightman维护的计算机视觉模型库，包含大量有监督和自监督预训练的视觉骨干网络（如ViT, ResNet, ConvNeXt）及其代码。

        * OpenAI CLIP

            * 链接: https://github.com/openai/CLIP

            * 介绍：官方实现的CLIP模型，可以轻松提取图文对齐的特征，进行零样本预测。

        * FAIR’s Detectron2 & D2 Go

            * 链接: https://github.com/facebookresearch/detectron2

            * 介绍：Meta AI的视觉项目，不仅包含检测分割模型，也集成了很多先进的视觉自监督学习算法（如MoCo v2, DINO）。

    2. 自监督学习专项库

        * Lightly

            * 链接: https://github.com/lightly-ai/lightly

            * 介绍：专注于自监督学习的轻量级PyTorch库，提供了SimCLR, MoCo, BYOL, DINO等多种算法的简洁实现和数据流水线。

        * VISSL

            * 链接: https://github.com/facebookresearch/vissl

            * 介绍：Meta AI发布的视觉自监督学习库，整合了其大量研究（SwAV, SimCLR, MoCo等），适合进行大规模研究和复现。

        * DINO (Self-Distillation with no labels)

            * 链接: https://github.com/facebookresearch/dino

            * 介绍：Meta AI提出的无标签知识蒸馏方法，能学习到优异的视觉表征，并展现出突出的物体分割特性。

    3. 图表征学习库

        * PyTorch Geometric (PyG)

            * 链接: https://github.com/pyg-team/pytorch_geometric

            * 介绍：最流行的图神经网络库之一，实现了大量GNN模型（包括图自编码器、对比学习模型），是学习图表征的绝佳工具。

        * Deep Graph Library (DGL)

            * 链接: https://www.dgl.ai/

            * 介绍：另一个高效、易用的图神经网络库，支持多种后端，有丰富的模型示例和教程。

    学习建议路径

    * 入门：从 Word2Vec 和 Autoencoder 理解基本思想，然后深入 BERT 和 SimCLR/MoCo 理解现代自监督范式。

    * 实践：使用 Hugging Face 和 timm 库，加载预训练模型，在自己的数据上提取特征或进行微调。

    * 深入：选择一个方向（如多模态CLIP、图对比学习），研读最新论文，并用 Lightly、PyG 等库进行代码复现和实验。

    * 跟踪前沿：关注顶会（NeurIPS, ICLR, CVPR, ACL）中“Representation Learning”、“Self-Supervised Learning”、“Multi-modal”等主题的论文。

    希望这份列表能为你探索表征学习的广阔天地提供一个有力的起点！

* 对于$z = w_1 x_1 + w_2  x_2$的情况，我们首先模仿只有一个$x$的写法，得到

    ```py
    def phi(z, theta):
        if z >= theta:
            return 1
        else:
            return -1
        
    def z(x1, x2, w1, w2):
        return w1 * x1 + w2 * x2

    def z_gt(x1, x2):
        return 0.3 * x1 + 0.5 * x2

    def main_3():
        theta = 0.5
        w1, w2 = 0.7, 0.2
        for i in range(10):
            x1 = random.random() * 2
            x2 = random.random() * 2
            val_z_gt = z_gt(x1, x2)
            val_z = z(x1, x2, w1, w2)
            val_phi_gt = phi(val_z_gt, theta)
            val_phi = phi(val_z, theta)
            if abs(val_phi_gt - val_phi) < 0.0001:
                delta_w1 = 0
                delta_w2 = 0
            elif val_phi_gt - val_phi > 0:
                delta_w1 = 0.1
                delta_w2 = 0.1
            else:
                delta_w1 = -0.1
                delta_w2 = -0.1
            print('round {}, w1: {:.2f}, w2: {:.2f}, delta_w1: {:.2f}, delta_w2: {:.2f}'.format(
                i, w1, w2, delta_w1, delta_w2))
            w1 += delta_w1
            w2 += delta_w2
        return

    if __name__ == '__main__':
        main_3()
    ```

    output:

    ```
    round 0, w1: 0.70, w2: 0.20, delta_w1: -0.10, delta_w2: -0.10
    round 1, w1: 0.60, w2: 0.10, delta_w1: 0.00, delta_w2: 0.00
    round 2, w1: 0.60, w2: 0.10, delta_w1: 0.00, delta_w2: 0.00
    round 3, w1: 0.60, w2: 0.10, delta_w1: 0.10, delta_w2: 0.10
    round 4, w1: 0.70, w2: 0.20, delta_w1: 0.00, delta_w2: 0.00
    round 5, w1: 0.70, w2: 0.20, delta_w1: -0.10, delta_w2: -0.10
    round 6, w1: 0.60, w2: 0.10, delta_w1: 0.00, delta_w2: 0.00
    round 7, w1: 0.60, w2: 0.10, delta_w1: 0.00, delta_w2: 0.00
    round 8, w1: 0.60, w2: 0.10, delta_w1: 0.00, delta_w2: 0.00
    round 9, w1: 0.60, w2: 0.10, delta_w1: 0.00, delta_w2: 0.00
    ```

    我们看到$w_1$与$w_2$绑定在了一起，要么都增大，要么都减小。还缺少一个关键因素，如果$w_1$偏大，$w_2$偏小，我们希望`delta_w1`为负数，`delta_w2`为正数。不清楚该如何做到。

    思路二，我们可以把$x$看作已知量，把$w$看作变量，此时只要求出$z$对$w$的偏导，即$x$，即可得到变化的方向。由此可写出如下代码：

    ```py
    def phi(z, theta):
        if z >= theta:
            return 1
        else:
            return -1
        
    def z(x1, x2, w1, w2):
        return w1 * x1 + w2 * x2

    def z_gt(x1, x2):
        return 0.3 * x1 + 0.5 * x2

    def main_3():
        theta = 0.5
        w1, w2 = 0.7, 0.2
        for i in range(10):
            x1 = random.random() * 2
            x2 = random.random() * 2
            val_z_gt = z_gt(x1, x2)
            val_z = z(x1, x2, w1, w2)
            val_phi_gt = phi(val_z_gt, theta)
            val_phi = phi(val_z, theta)
            if abs(val_phi_gt - val_phi) < 0.0001:
                delta_w1 = 0
                delta_w2 = 0
            elif val_phi_gt - val_phi > 0:
                delta_w1 = 0.1 * x1
                delta_w2 = 0.1 * x2
            else:
                delta_w1 = -0.1 * x1
                delta_w2 = -0.1 * x2
            print('round {}, w1: {:.2f}, w2: {:.2f}, delta_w1: {:.2f}, delta_w2: {:.2f}'.format(
                i, w1, w2, delta_w1, delta_w2))
            w1 += delta_w1
            w2 += delta_w2
        return

    if __name__ == '__main__':
        main_3()
    ```

    output:

    ```
    round 0, w1: 0.70, w2: 0.20, delta_w1: 0.00, delta_w2: 0.00
    round 1, w1: 0.70, w2: 0.20, delta_w1: 0.00, delta_w2: 0.00
    round 2, w1: 0.70, w2: 0.20, delta_w1: 0.00, delta_w2: 0.00
    round 3, w1: 0.70, w2: 0.20, delta_w1: 0.00, delta_w2: 0.00
    round 4, w1: 0.70, w2: 0.20, delta_w1: -0.09, delta_w2: -0.02
    round 5, w1: 0.61, w2: 0.18, delta_w1: 0.00, delta_w2: 0.00
    round 6, w1: 0.61, w2: 0.18, delta_w1: 0.00, delta_w2: 0.18
    round 7, w1: 0.61, w2: 0.36, delta_w1: 0.00, delta_w2: 0.00
    round 8, w1: 0.61, w2: 0.36, delta_w1: 0.00, delta_w2: 0.00
    round 9, w1: 0.61, w2: 0.36, delta_w1: 0.00, delta_w2: 0.00
    ```

    这个方向明显是对的。

    这里借助了偏导。如果不知道偏导数这个概念，是否可以用其他方式想出来？

* 画$y = \begin{cases} 1,\ x \geq \theta \\ -1,\ x \lt \theta \end{cases}$的图像，当$\theta = 0.5$时，有

    <div>
    <img width=700 src='../../Reference_resources/ref_35/pic_1.png'>
    </div>

    这里想讨论的是，这个函数是否需要能够微分？

    我们需要的似乎只是$y_{gt} - y$的正负，不需要知道这个函数的导数。

* 乘法的大与小

    对于$z = xy$，如果想增大$z$，那么当$x \gt 0$时，可以增大$y$；当$x \lt 0$时，可以减小$y$。如果想减小$z$，那么当$x \gt 0$时，需要减小$y$，当$x \lt 0$时，需要增大$y$。

    另外一种想法是，我们把$x$与$0$的大小比值提前，当$x \gt 0$，如果想增大$z$，那么需要增大$y$；如果想减小$z$，那么需要减小$y$。当$x \gt 0$，如果想增大$z$，那么需要减小$y$，如果想减小$z$，则需要增大$y$。

    现在的问题是：这样的分类讨论是否有更简洁统一的表述？

* 离散情况下通过负反馈复合函数的参数

    我们考虑最简单的一种情况：$z = wx$，$\phi(z) = \begin{cases} 1, \ z >= \theta \\ -1, \ z < \theta \end{cases}$，为简单起见，我们令$\theta = 0.5$。现在假如$w_{\mathrm{gt}} = 0.7$，我们猜测$w = 0.5$，并且我们只能计算$\phi(x)$，那么是否存在一个算法，可以使我们的猜测值$w$不断逼近真实值$w_{gt}$？

    想法：我们对$x$随机采样，并计算$\phi_{\mathrm{gt}}(x)$和$\phi(x)$，若$\phi_{\mathrm{gt}} - \phi \gt 0$，则说明$w$偏小。如果我们能证明前面这句话，那么就可以写出下面的算法：

    ```python
    import random
    def main_3():
        w = 0.5
        w_gt = 0.7
        theta = 0.5
        for i in range(10):
            x = random.random() * 1.5  # [0, 1.5)
            z = w * x
            z_gt = w_gt * x
            phi = 1 if z > theta else -1
            phi_gt = 1 if z_gt > theta else -1
            w_delta = 0.1 if phi_gt - phi > 0 else -0.1
            if abs(phi_gt - phi) < 0.0001:
                w_delta = 0
            elif phi_gt - phi > 0:
                w_delta = 0.1
            else:
                w_delta = -0.1
            print('round {}, x: {:.4}, phi: {}, phi_gt: {}, w: {}, w_delta: {}'.format(
                i, x, phi, phi_gt, w, w_delta)
            )
            w += w_delta
        return

    if __name__ == '__main__':
        main_3()
    ```

    output:

    ```
    round 0, x: 0.9404, phi: -1, phi_gt: 1, w: 0.5, w_delta: 0.1
    round 1, x: 0.9234, phi: 1, phi_gt: 1, w: 0.6, w_delta: 0
    round 2, x: 0.8939, phi: 1, phi_gt: 1, w: 0.6, w_delta: 0
    round 3, x: 0.7078, phi: -1, phi_gt: -1, w: 0.6, w_delta: 0
    round 4, x: 0.019, phi: -1, phi_gt: -1, w: 0.6, w_delta: 0
    round 5, x: 0.8664, phi: 1, phi_gt: 1, w: 0.6, w_delta: 0
    round 6, x: 0.9605, phi: 1, phi_gt: 1, w: 0.6, w_delta: 0
    round 7, x: 0.5712, phi: -1, phi_gt: -1, w: 0.6, w_delta: 0
    round 8, x: 0.8324, phi: -1, phi_gt: 1, w: 0.6, w_delta: 0.1
    round 9, x: 0.4142, phi: -1, phi_gt: -1, w: 0.7, w_delta: 0
    ```

    从结果看，我们的假设很可能是对的。

    尝试分析一下：

    我们画出$\phi(z)$和$\phi_{\mathrm{gt}}(z)$的图像：

    <div>
    <img alt='phi(z)' width=700 src=''>
    </div>

    再画出$z(x)$和$z_{\mathrm{gt}}(x)$的图像：

    <div>
    <img alt='z(x)' width=700 src=''>
    </div>

    可见在$(x_0, x_1)$范围内，$\phi_{\mathrm{gt}}$总是$1$，$\phi$总是$-1$，此时会不断增大$w$，并且在$w \lt w_{\mathrm{gt}}$时，$\phi_{\mathrm{gt}} \gt \phi$总是成立。

    说明：

    1. 这个过程没有计算导数，只是一个负反馈通路，但是需要全程白盒，知道当某种现象发生时，如何调整可调参数。$\phi(z)$就是为了防止计算导数。

    2. 现在考虑$z(x_1, x_2) = w_1 x_1 + w_2 x_2$，上述方案是否仍然可行？

    3. 这样分析似乎也可以：对于某个$x$，当$\phi_{gt} > \phi$时，我们需要增大$\phi$，因此需要增大$z$，因为$x$是正值，因此需要增大$w$。

* 小宽度 + 小深度时，l1 loss 下的拟合任务，sigmoid 的效果比 relu 更好

    当使用 sigmoid + 大深度时，拟合曲线会经常在局部有高频 sin 曲线的形状。

    如果每次随机 fix 3/4 的参数，只让 1/4 的参数被 optimizer 改变，在有限的 epoch 内是否效果会更好？或者每次只改变一个 weight。假设：当节点数量变多时，每个节点贡献的 effort 很小，但是也有一些是互相抵消的，是否可以让单个节点在限制条件下（其他节点参数不动），快速调整，找到自己的位置，从而使得单个节点的 contribute 最大？这样的话，就不存在互相抵消的影响了。如果不行，那么说明小 effort 比抵消作用更加重要，这样就要求我们去关注 init random seed，随机搜索的种子采样点。

* python 机器学习上感知器的例子

    ```python
    import numpy as np

    class Perceptron(object):
        def __init__(self, eta=0.01, n_iter=10):
            self.eta = eta
            self.n_iter = n_iter

        def fit(self, X, y):
            self.w_ = np.zeros(1 + X.shape[1])
            self.errors_ = []

            for _ in range(self.n_iter):
                errors = 0
                for xi, target in zip(X, y):
                    update = self.eta * (target - self.predict(xi))
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    errors += int(update != 0.0)
                self.errors_.append(errors)
            return self
        
        def net_input(self, X):
            return np.dot(X, self.w_[1:]) + self.w_[0]
        
        def predict(self, X):
            return np.where(self.net_input(X) >= 0.0, 1, -1)
    ```

* 感知器的学习算法

    1. 将权重$\boldsymbol w$初始化为零或一个极小的随机数

    2. 迭代所有的训练样本$\boldsymbol x^{(i)}$，执行以下操作：

        1. 计算输出值$\hat y$

        2. 更新权重

            $$w_j \leftarrow w_j + \Delta w_j$$

            $$\Delta w_j = \eta (y^{(i)} - \hat y^{(i)}) x_j^{(i)}$$

            其中，$\eta$是学习率，$0.0 \lt \eta \lt 1.0$。

* a simple perceptron function implemented by numpy

    ```python
    import numpy as np

    def perceptron(x: np.ndarray, w: np.ndarray, theta: float) -> float:
        z = w.T @ x
        z = z.item()  # convert z to python 'float' type
        print('z = %f' % z)
        if z >= theta:
            y = 1
        else:
            y = -1
        return y

    def main():
        w = np.array([0.3, 0.7])
        w = w.reshape(2, 1)
        x = np.array([0.5, 0.6])
        w = w.reshape(2, 1)
        theta = 0.5
        y = perceptron(w, x, theta)
        print('y = %f' % y)

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    z = 0.570000
    y = 1.000000
    ```

* 感知器算法

    设输入$\boldsymbol x = \begin{bmatrix} x_1 \\ \vdots \\ x_m \end{bmatrix}$，权重向量$\boldsymbol w = \begin{bmatrix} w_1 \\ \vdots \\ w_m \end{bmatrix}$，净输入$\boldsymbol z = w_1 x_1 + w_2 x_ 2 + \cdots + w_m x_m$，其实就是$z = \boldsymbol w^\intercal \boldsymbol x$
    
    定义激励函数为

    $$\phi (z) = \left\{ \begin{aligned} 1,\ &\text{if } z \geq \theta \\ -1,\ &\text{otherwise}  \end{aligned} \right.$$

    说明：

    1. 这里呈现的是多个输入神经元的输出通过乘一个权重后，累加作用于后置神经元。

        为什么这里刚好选择了乘权重？乘权重实际止是一个点乘操作，点乘正好对应了线性空间中的模式匹配准则。

* 术语

    感知器 perceptron

    自适应线性单元 adaptive linear neuron

    激励函数 activation function

    向量点积 vector dot product

    净输入 net input，这个指的就是$z = \boldsymbol w^\intercal \boldsymbol x$

* 鸢尾花数据集是事先已经知道了不同的类别，然后再在不同的类别上挑选了一些**主观**上认为有用的几个特征，再根据这些特征去训练模型。

    目前 AI 的难题是：事先不知道是否为相同的类别，也不知道选哪些特征可能有效。只能不断尝试收集新特征，直到突然发现有一个特征可以将一批数据集清晰地分类，那么就认为发现了一条新的“规律”。

* 鸢尾花数据集（Iris dataset）

    它包含了 Setosa、Versicolor 和 Virginica 三个品种总共 150 种鸢尾花的测量数据。

    数据表头：

    萼片长度，萼片宽度，花瓣长度，花瓣宽度，类标

    我们记此数据集为

    $$
    \bold X \in \mathbb R^{150 \times 4} = 
    \begin{bmatrix}
    x^{(1)}_1 &x^{(2)}_2 &x^{(3)}_3 &x^{(4)}_4 \\
    x^{(2)}_1 &x^{(2)}_2 &x^{(3)}_3 &x^{(4)}_4 \\
    \vdots &\vdots &\vdots &\vdots \\
    x^{(150)}_1 &x^{(150)}_2 &x^{(150)}_3 &x^{(150)}_4
    \end{bmatrix}
    $$

    记目标变量为

    $$
    \boldsymbol y =
    \begin{bmatrix}
    y^{(1)} \\
    \vdots \\
    y^{(150)}
    \end{bmatrix}
    (y \in \{ \text{Setosa, Versicolor, Virginica} \})
    $$

* 术语

    监督学习 supervised learning

    无监督学习 unsupervised learning

    强化学习 reinforcement learning

    分类 classification

    回归 regression

    样本的组别信息 group membership  (不知道是啥意思)

    二分类 binary classification

    多分类 multi-class classification

    负类别 negative class

    正类别 positve class

    线性回归 linear regression

    聚类中的簇 cluster

    数据降维 dimensionality reduction

## note

(empty)
