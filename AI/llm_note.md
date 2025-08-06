# llm note

## cache

* Yoshua Bengio（约书亚·本吉奥）

    核心方向：

    1. 贝叶斯神经网络（Bayesian Neural Networks, BNN）

    2. 生成模型与贝叶斯框架

        * 贝叶斯生成对抗网络（Bayesian GAN），通过贝叶斯方法优化生成器和判别器的参数分布，提升生成样本的多样性和稳定性。

        * 变分自编码器（VAE）

            通过变分推断学习隐变量的后验分布

    3. 注意力机制中的贝叶斯视角

        在Transformer和注意力机制中，Bengio团队尝试用贝叶斯方法建模注意力权重的不确定性，例如通过随机注意力（Stochastic Attention）提升鲁棒性。

    4. 因果推断与贝叶斯网络

        为贝叶斯网络（Bayesian Networks）和神经网络结合可解决AI的因果表征学习问题

* decoder-only arch

    由解码器堆叠而成。
    
    单向注意力，仅允许每个位置关注左侧历史信息（掩码自注意力）。每次只预测下一个 token。

    gpt 系列就是 decoder-only 架构。

    自回归（逐token生成），确保输出连贯性，但Encoder-Decoder通过编码器的双向建模能更好理解输入，适合复杂映射任务。Decoder-Only因结构简单，在大规模预训练中更高效。

    左侧历史信息 (left-context) : 在生成第$t$个 token 时，模型只能看到当前位置之前的所有 token（即位置 $1$, $2$, $...$, $t−1$），而无法访问未来的 token（位置 $t+1$, $t+2$, $...$）。这样保证生成过程的自回归性（逐词生成），符合语言的自然顺序

## note