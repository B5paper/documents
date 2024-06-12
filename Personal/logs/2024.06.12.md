* attention is all you need note

    这篇文章提出了 transformer，效率更高一些，在翻译任务上取得了 sota。

    这篇文章提出的背景是 RNN, LSTM，GRNN

    这个时期 encoder 和 decoder 的概念已经被提出来了。

    可能用到的 tricks: 
    
    1. factorization tricks

    2. conditional computation

    对于 ConvS2S 和 ByteNet，他们采用卷积来抽取特征层，但是由于卷积没法快速将两个远距离的特征联系到一起，可能需要多步卷积才能做到，因此他们的效率不高。

    拥有 encoder 和 decoder 结构的序列处理神经网络，有空了可以调研一下：

    1. Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.

    2. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.

    3. Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104–3112, 2014.

    auto-regressive 自回归的概念，有时间了看下：Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.

    transformer 由 N 个 blocks 搭建而成，每个 block 又有残差连接，残差的输出还加了一层 layer norm。每个 layer 的输出维度为 512，因为 layer 中主要是 fully connected 结构，所以可以认为输出其实是 512 个量。

    decoder 由 6 个相同的 layer 构成，每个 layer 除了有 2 个和 encoder layer 相同的 sublayer，还有 1 个额外的 multi-head layer。

    mask 的作用似乎是让 output 只和输入的数据相关。

    这篇文章的 self-attension 层被作者称为 scaled dot-product attention。其接收两个输入，query,key 。query, key 都是一个$d_k$行，不知道多少列（假设为$n$）的矩阵，value 是一个$n$行，$d_v$列的矩阵。

    于是我们可以将 self-attention 层的公式写为：

    $$Attention(Q, K, T) = softmax(\dfrac{Q \cdot K^T}{\sqrt{d_k}}) \cdot V$$

    矩阵乘法实现的注意力效率要比加法实现的注意力更高。

    每个 attention head 只能获得线性注意力，因此使用多个（论文中是 8 个） head 获得多个注意力，提升精度。对于每个 head，$Q$，$K$，$V$分别被乘一个矩阵$W_i^Q$，$W_i^K$，$W_i^V$。目的是投影到不同的注意力通路上。

    sequence-to-sequence model: 31, 2, 8

    position-wise fully connection 指的是什么意思？为什么需要 position-wise？position-wise 为什么会和全连接层扯上关系？这个全连接层可以被 1x1 的卷积层替代。全连接层的 shape 为 512 -> 2048 -> 512。

    由于模型里不包含循环（recurrence）也不包含卷积，因此需要一种方法确定 token 的绝对位置（absolute position）。transformer 使用$\sin$和$\cos$对 token 的位置进行编码。