# LLMs 千面郎君

> 介绍：本项目是作者们根据个人面试和经验总结出的 大模型(LLMs)面试准备的学习笔记与资料，该资料目前包含 大模型(LLMs)各领域的 面试题积累。

<img src="img/微信截图_20230918094559.png" width="50%" >
> LLMs 千面郎君 面试交流群 (注：人满 可 添加 小编wx：yzyykm666 加群！)

<img src="img/微信截图_20210301212242.png" width="50%" >

## [大模型（LLMs）基础面](https://articles.zsxq.com/id_a55uo10835nv.html)

1. 目前 主流的开源模型体系 有哪些？
2. prefix Decoder 和 causal Decoder 和 Encoder-Decoder 区别是什么？
3. 大模型LLM的 训练目标 是什么？
4. 涌现能力是啥原因？
5. 为何现在的大模型大部分是Decoder only结构？
6. 简单 介绍一下 大模型【LLMs】？
7. 大模型【LLMs】后面跟的 175B、60B、540B等 指什么？
8. 大模型【LLMs】具有什么优点？
9. 大模型【LLMs】具有什么缺点？
10. ...

- [点击查看答案](https://articles.zsxq.com/id_a55uo10835nv.html)

## [大模型（LLMs）进阶面](https://articles.zsxq.com/id_v6gltxd4qbxd.html)

1. LLMs 复读机问题
   1. 什么是 LLMs 复读机问题？
   2. 为什么会出现 LLMs 复读机问题？
   3. 如何缓解 LLMs 复读机问题？
2. llama 系列问题
   1. llama 输入句子长度理论上可以无限长吗？
3. 什么情况用Bert模型，什么情况用LLaMA、ChatGLM类大模型，咋选？
4. 各个专业领域是否需要各自的大模型来服务？
5. 如何让大模型处理更长的文本？
6. ...

- [点击查看答案](https://articles.zsxq.com/id_v6gltxd4qbxd.html)

## [大模型（LLMs）微调面](https://articles.zsxq.com/id_khze6sgassi3.html)

### [大模型（LLMs）微调面](https://articles.zsxq.com/id_khze6sgassi3.html)

1. 如果想要在某个模型基础上做全参数微调，究竟需要多少显存？
2. 为什么SFT之后感觉LLM傻了?
3. SFT 指令微调数据 如何构建?
4. 领域模型Continue PreTrain 数据选取？
5. 领域数据训练后，通用能力往往会有所下降，如何缓解模型遗忘通用能力？
6. 领域模型Continue PreTrain ，如何 让模型在预训练过程中就学习到更多的知识？
7. 进行SFT操作的时候，基座模型选用Chat还是Base?
8. 领域模型微调 指令\&数据输入格式 要求？
9. 领域模型微调 领域评测集 构建？
10. 领域模型词表扩增是不是有必要的？
11. 如何训练自己的大模型？
12. 训练中文大模型有啥经验？
13. 指令微调的好处？
14. 预训练和微调哪个阶段注入知识的？
15. 想让模型学习某个领域或行业的知识，是应该预训练还是应该微调？
16. 多轮对话任务如何微调模型？
17. 微调后的模型出现能力劣化，灾难性遗忘是怎么回事？
18. 微调模型需要多大显存？
19. 大模型LLM进行SFT操作的时候在学习什么？
20. 预训练和SFT操作有什么不同
21. 样本量规模增大，训练出现OOM错
22. 大模型LLM进行SFT 如何对样本进行优化？
23. 模型参数迭代实验
24. 微调大模型的一些建议
25. ...

- [点击查看答案](https://articles.zsxq.com/id_khze6sgassi3.html)

### [大模型（LLMs）训练经验帖](https://articles.zsxq.com/id_06n25d9wjs0e.html)

- 分布式训练框架选择？
- LLMs 训练时 有哪些有用的建议？
- 模型大小如何选择？
- 加速卡如何选择？
- ...

- [点击查看答案](https://articles.zsxq.com/id_06n25d9wjs0e.html)

## 大模型（LLMs）langchain 面

### [大模型（LLMs）langchain 面](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

- 1. 什么是 LangChain?
- 2. LangChain 包含哪些 核心概念？
  - 2.1 LangChain 中 Components and Chains 是什么？
  - 2.2 LangChain 中 Prompt Templates and Values 是什么？
  - 2.3 LangChain 中 Example Selectors 是什么？
  - 2.4 LangChain 中 Output Parsers 是什么？
  - 2.5 LangChain 中 Indexes and Retrievers 是什么？
  - 2.6 LangChain 中  Chat Message History 是什么？
  - 2.7 LangChain 中  Agents and Toolkits 是什么？
- 3. 什么是 LangChain Agent?
- 4. 如何使用 LangChain ?
- 5. LangChain 支持哪些功能?
- 6. 什么是 LangChain model?
- 7. LangChain 包含哪些特点?
- 8. LangChain 如何使用?
  - 8.1 LangChain 如何调用 LLMs 生成回复？
  - 8.2 LangChain 如何修改 提示模板？
  - 8.3 LangChain 如何链接多个组件处理一个特定的下游任务？
  - 8.4 LangChain 如何Embedding \& vector store？
- LangChain 存在哪些问题及方法方案？
  - 1. LangChain 低效的令牌使用问题
  - 2. LangChain 文档的问题
  - 3. LangChain 太多概念容易混淆，过多的“辅助”函数问题
  - 4. LangChain 行为不一致并且隐藏细节问题
  - 5. LangChain 缺乏标准的可互操作数据类型问题
- LangChain 替代方案？
- ...

- [点击查看答案](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### [基于LLM+向量库的文档对话 经验面](https://articles.zsxq.com/id_m9t1w8pokjpf.html)

- 一、基于LLM+向量库的文档对话 基础面
  - 1.1 为什么 大模型 需要 外挂(向量)知识库？
  - 1.2. 基于LLM+向量库的文档对话 思路是怎么样？
  - 1.3. 基于LLM+向量库的文档对话 核心技术是什么？
  - 1.4. 基于LLM+向量库的文档对话 prompt 模板 如何构建？
- 二、基于LLM+向量库的文档对话 存在哪些痛点？
- 三、基于LLM+向量库的文档对话 工程示例面
- ...

- [点击查看答案](https://articles.zsxq.com/id_m9t1w8pokjpf.html)

### [LLM文档对话 —— pdf解析关键问题](https://articles.zsxq.com/id_2693k55it84w.html)

- 一、为什么需要进行pdf解析？
- 二、为什么需要 对 pdf 进行解析？
- 三、pdf解析 有哪些方法，对应的区别是什么？
- 四、pdf解析 存在哪些问题？
- 五、如何 长文档（书籍）中关键信息？
- 六、为什么要提取标题甚至是多级标题？
- 七、如何提取 文章标题？
- 八、如何区分单栏还是双栏pdf？如何重新排序？
- 九、如何提取表格和图片中的数据？
- 十、基于AI的文档解析有什么优缺点？
- ...

- [点击查看答案](https://articles.zsxq.com/id_2693k55it84w.html)

### [基于LLM+向量库的文档对话 经验面](https://articles.zsxq.com/id_m9t1w8pokjpf.html)

- 一、基于LLM+向量库的文档对话 基础面
  - 1.1 为什么 大模型 需要 外挂(向量)知识库？
  - 1.2. 基于LLM+向量库的文档对话 思路是怎么样？
  - 1.3. 基于LLM+向量库的文档对话 核心技术是什么？
  - 1.4. 基于LLM+向量库的文档对话 prompt 模板 如何构建？
- 二、基于LLM+向量库的文档对话 存在哪些痛点？
- 三、基于LLM+向量库的文档对话 工程示例面
- ...

- [点击查看答案](https://articles.zsxq.com/id_m9t1w8pokjpf.html)


## [大模型（LLMs）参数高效微调(PEFT) 面](https://articles.zsxq.com/id_ahk2br3igwx9.html)

### [大模型（LLMs）参数高效微调(PEFT) 面](https://articles.zsxq.com/id_ipkod91a939n.html)

- 微调方法是啥？如何微调？
- 为什么需要 PEFT？
- 介绍一下 PEFT？
- PEFT 有什么优点？
- 微调方法批处理大小模式GPU显存速度？
- Peft 和 全量微调区别？
- 多种不同的高效微调方法对比
- 当前高效微调技术存在的一些问题
- 高效微调技术最佳实践
- PEFT 存在问题？
- 能不能总结一下各种参数高效微调方法？
- ...

- [点击查看答案](https://articles.zsxq.com/id_ipkod91a939n.html)

### [配器微调（Adapter-tuning）篇](https://articles.zsxq.com/id_h5q2fzq8wvt8.html)

- 一、为什么 需要 适配器微调（Adapter-tuning）？
- 二、适配器微调（Adapter-tuning）思路？
- 三、 适配器微调（Adapter-tuning）特点是什么？
- 四、AdapterFusion 思路 是什么？
- 五、AdapterDrop 思路 是什么？
- 六、AdapterDrop 特点 是什么？
- 七、MAM Adapter 思路 是什么？
- 八、MAM Adapter 特点 是什么？
- ...

- [点击查看答案](https://articles.zsxq.com/id_h5q2fzq8wvt8.html)

### [提示学习（Prompting）](https://articles.zsxq.com/id_662wpbw47gtj.html)

- 一、为什么需要 提示学习（Prompting）？
- 二、什么是 提示学习（Prompting）？
- 三、提示学习（Prompting） 有什么优点？
- 四、提示学习（Prompting）有哪些方法，能不能稍微介绍一下它们间？
  - 4.1 前缀微调（Prefix-tining）篇
    - 4.1.1 为什么需要 前缀微调（Prefix-tining）？
    - 4.1.2 前缀微调（Prefix-tining）思路是什么？
    - 4.1.3 前缀微调（Prefix-tining）的优点是什么？
    - 4.1.4 前缀微调（Prefix-tining）的缺点是什么？
  - 4.2 指示微调（Prompt-tuning）篇
    - 4.2.1 为什么需要 指示微调（Prompt-tuning）？
    - 4.2.2 指示微调（Prompt-tuning）思路是什么？
    - 4.2.3 指示微调（Prompt-tuning）优点是什么？
    - 4.2.4 指示微调（Prompt-tuning）缺点是什么？
    - 4.2.5 指示微调（Prompt-tuning）与 Prefix-tuning 区别 是什么？
    - 4.2.6 指示微调（Prompt-tuning）与 fine-tuning 区别 是什么？
  - 4.3 P-tuning 篇
    - 4.3.1 为什么需要 P-tuning？
    - 4.3.2 P-tuning 思路是什么？
    - 4.3.3 P-tuning 优点是什么？
    - 4.3.4 P-tuning 缺点是什么？
  - 4.4 P-tuning v2 篇
    - 4.4.1 为什么需要 P-tuning v2？
    - 4.4.2 P-tuning v2 思路是什么？
    - 4.4.3 P-tuning v2 优点是什么？
    - 4.4.4 P-tuning v2 缺点是什么？
- ...

- [点击查看答案](https://articles.zsxq.com/id_662wpbw47gtj.html)
  
### [LoRA 系列篇](https://articles.zsxq.com/id_ham28l44907e.html)

- 一、LoRA篇
  - 1.1 什么是 LoRA？
  - 1.2 LoRA 的思路是什么？
  - 1.3 LoRA 的特点是什么？
- 二、QLoRA篇
  - 2.1 QLoRA 的思路是怎么样的？
  - 2.2 QLoRA 的特点是什么？
- 三、AdaLoRA篇
  - 3.1 AdaLoRA 的思路是怎么样的？
- 四、LoRA权重是否可以合入原模型？
- 五、ChatGLM-6B LoRA后的权重多大？
- 六、LoRA 微调优点是什么？
- 七、LoRA微调方法为啥能加速训练？
- 八、如何在已有LoRA模型上继续训练？
- 九、LoRA 缺点是什么？
- 十、LoRA这种微调方法和全参数比起来有什么劣势吗？
- ...

- [点击查看答案](https://articles.zsxq.com/id_ham28l44907e.html)

## [大模型（LLMs）推理面](https://articles.zsxq.com/id_udwh2i8seqv8.html)

### [大模型（LLMs）推理面](https://articles.zsxq.com/id_udwh2i8seqv8.html)

- 1. 为什么大模型推理时显存涨的那么多还一直占着？
- 2. 大模型在gpu和cpu上推理速度如何？
- 3. 推理速度上，int8和fp16比起来怎么样？
- 4. 大模型有推理能力吗？
- 5. 大模型生成时的参数怎么设置？
- 6. 有哪些省内存的大语言模型训练/微调/推理方法？
  - 6.1 如何 估算模型所需的RAM？
  - 6.2 Fp16-mixed precision
  - 6.3 Int8-bitsandbytes
  - 6.4 LoRA
  - 6.5 Gradient Checkpointing
  - 6.6 Torch FSDP+CPU offload
- 7. 如何让大模型输出合规化
- 8. 应用模式变更
- ...

- [点击查看答案](https://articles.zsxq.com/id_udwh2i8seqv8.html)

## 大模型（LLMs）预训练面

### [大模型（LLMs）增量预训练篇](https://articles.zsxq.com/id_lj47ancwcmv2.html)

1. 为什么要增量预训练？
2. 进行 增量预训练 需要做哪些准备工作？
3. 增量预训练 所用 训练框架？
4. 增量预训练 训练流程 是怎么样？
5. ...

- [点击查看答案](https://articles.zsxq.com/id_lj47ancwcmv2.html)

## [大模型（LLMs）评测面](https://articles.zsxq.com/id_j9wcj62eovgc.html)

1. 大模型怎么评测？
2. 大模型的honest原则是如何实现的？模型如何判断回答的知识是训练过的已知的知识，怎么训练这种能力？
3. 如何衡量大模型水平？
4. 大模型评估方法 有哪些？
5. 大模型评估工具 有哪些？
6. ...

- [点击查看答案](https://articles.zsxq.com/id_j9wcj62eovgc.html)

## [大模型（LLMs）强化学习面](https://articles.zsxq.com/id_zqs7mjw6c8k7.html)

- 1. 简单介绍强化学习？
- 2. 简单介绍一下 RLHF？
- 3. 奖励模型需要和基础模型一致吗？
- 4. RLHF 在实践过程中存在哪些不足？
- 5. 如何解决 人工产生的偏好数据集成本较高，很难量产问题？
- 6. 如何解决三个阶段的训练（SFT-\>RM-\>PPO）过程较长，更新迭代较慢问题？
- 7. 如何解决 PPO 的训练过程同时存在4个模型（2训练，2推理），对计算资源的要求较高 问题？
- ...

- [点击查看答案](https://articles.zsxq.com/id_zqs7mjw6c8k7.html)

## [大模型（LLMs）软硬件配置面](https://articles.zsxq.com/id_m5q8zk3wo84k.html)

1. 建议的软件环境是什么？
2. ...

- [点击查看答案](https://articles.zsxq.com/id_m5q8zk3wo84k.html)

## [大模型（LLMs）训练集面](https://articles.zsxq.com/id_jwvpaujrojtt.html)

1. SFT（有监督微调）的数据集格式？
2. RM（奖励模型）的数据格式？
3. PPO（强化学习）的数据格式？
4. 找数据集哪里找？
5. 微调需要多少条数据？
6. 有哪些大模型的训练集？
7. 进行领域大模型预训练应用哪些数据集比较好？
8. ...

- [点击查看答案](https://articles.zsxq.com/id_jwvpaujrojtt.html)

## [大模型（LLMs）显存问题面](https://articles.zsxq.com/id_jhiocx89p3su.html)

1. 大模型大概有多大，模型文件有多大?
2. 能否用4 * v100 32G训练vicuna 65b？
3. 如果就是想要试试65b模型，但是显存不多怎么办？
4. nB模型推理需要多少显存？
5. nB模型训练需要多少显存？
6. 如何 估算模型所需的RAM？
7. 如何评估你的显卡利用率?
8. 测试你的显卡利用率 实现细节篇
   1. 如何查看多机训练时的网速？
   2. 如何查看服务器上的多卡之间的NVLINK topo？
   3. 如何查看服务器上显卡的具体型号?
   4. 如何查看训练时的flops？（也就是每秒的计算量）
   5. 如何查看对deepspeed的环境配置是否正确？
   6. tf32格式有多长？
   7. 哪里看各类显卡算力比较？
   8. （torch profiler）如何查看自己的训练中通信开销？
- ...

- [点击查看答案](https://articles.zsxq.com/id_jhiocx89p3su.html)

## 大模型（LLMs）分布式训练面

### [大模型（LLMs）分布式训练面](https://articles.zsxq.com/id_ah2ibj3z22c7.html)

- 1. 理论篇
  - 1.1 训练 大语言模型 存在问题？
  - 1.2 什么是 点对点通信？
  - 1.3 什么是 集体通信？
  - 1.4 什么是 数据并行？
  - 1.5 数据并行 如何 提升效率？
  - 1.6 什么是 流水线并行？
  - 1.7 什么是 张量并行 (intra-layer)？
  - 1.8 数据并行 vs 张量并行 vs 流水线并行?
  - ...
- 2. 实践篇
  - 2.1 假如有超多的8卡A100节点（DGX A100），如何应用3D并行策略？
  - 2.2 如果想构这样一个大规模并行训练系统，训练框架如何选？
  - 2.3 训练框架如何选？
- 3. 并行化策略选择篇
  - 3.1 如何选择一款分布式训练框架？
  - 3.2 如何选择一款分布式训练框架？
  - 3.3 单GPU
  - 3.4 单节点多卡
  - 3.5 多节点多卡
- 4. 问题篇
  - 4.1 推理速度验证
  - 4.2 并行化训练加速
  - 4.3 deepspeed 训练过程，报找不主机
  - 4.4 为什么 多机训练效率不如单机？
  - 4.5 多机训练不通，DeepSPeed配置问题
- ...

- [点击查看答案](https://articles.zsxq.com/id_ah2ibj3z22c7.html)

### [图解分布式训练（一） —— 流水线并行（Pipeline Parallelism）面](https://articles.zsxq.com/id_ah2ibj3z22c7.html)

- 为什么需要流水线并行（Pipeline Parallelism）？
- 一、流水线并行（Pipeline Parallelism） 优化目标是什么？
- 二、图解 流水线并行（Pipeline Parallelism）模型并行 必要性？
- 三、流水线并行（Pipeline Parallelism） 图解？
- 四、流水线并行（Pipeline Parallelism）优缺点？
- ...

- [点击查看答案](https://articles.zsxq.com/id_wre1eni0oq7d.html)

### [图解分布式训练（二） —— nn.DataParallel面](https://articles.zsxq.com/id_ah2ibj3z22c7.html)

- 为什么需要nn.DataParallel？
- 一、pytorch中的GPU操作默认是什么样？
- 二、介绍一下 nn.DataParallel 函数？
- 三、nn.DataParallel 函数 处理逻辑 介绍一下？
- 四、nn.DataParallel 函数 常见问题及解答 有哪些？
  - 4.1 多GPU计算减少了程序运行的时间？
  - 4.2 如何保存和加载多GPU训练模型呢？
  - 4.3 为什么第一块卡的显存会占用的更多一些？
  - 4.4 直接使用nn.DataParallel的时候，训练采用多卡训练，会出现一个warning？
  - 4.5 device\_ids 0 被占用问题
- 五、nn.DataParallel 函数 参数更新方式 ？
- 六、nn.DataParallel 函数 优点 介绍一下？
- 七、nn.DataParallel 函数 缺点 介绍一下？
- 八、nn.DataParallel 函数 实战？
- ...

- [点击查看答案](https://articles.zsxq.com/id_wre1eni0oq7d.html)

### [图解分布式训练（三） ——  nn.parallel.DistributedDataParallel](https://articles.zsxq.com/id_i4s3ia057rmh.html)

- 为什么需要 nn.parallel.DistributedDataParallel ？
- 一、什么是 DistributedDataParallel 核心 —— Ring-AllReduce？
- 二、nn.parallel.DistributedDataParallel 函数 介绍一下？
- 三、nn.parallel.DistributedDataParallel 函数 如何多卡加速训练？
- 四、nn.parallel.DistributedDataParallel 实现流程介绍一下？
- 五、nn.parallel.DistributedDataParallel 参数更新介绍一下？
- 六、nn.DataParallel(以下简称DP) vs DistributedDataParallel(以下简称DDP)介绍一下？
- 七、DistributedDataParallel(以下简称DDP) 优点有哪些？
- 八、DistributedDataParallel(以下简称DDP) 缺点有哪些？
- ...

- [点击查看答案](https://articles.zsxq.com/id_i4s3ia057rmh.html)

### [图解分布式训练（四） ——  torch.multiprocessing 详细解析](https://articles.zsxq.com/id_gu9smpbn510e.html)

- 一、torch.multiprocessing 函数介绍一下？
- 二、torch.multiprocessing 函数如何使用？
- 三、介绍一下 共享CUDA张量？
- 四、介绍一下 共享策略？
- 五、torch.multiprocessing 函数使用
- ...

- [点击查看答案](https://articles.zsxq.com/id_gu9smpbn510e.html)

### [图解分布式训练（五） ——  AMP混合精度训练 详细解析](https://articles.zsxq.com/id_0slrgoti6gvb.html)

- 为什么需要 AMP混合精度训练？
- 一、什么是自动混合精度训练(AMP)
- 二、为什么需要自动混合精度？
- 三、混合精度训练的优点是什么？
- 四、混合精度训练的缺点是什么？
- 五、混合精度训练的关键技术是什么？
- 六、介绍一下 混合精度训练 动态损失缩放？
- 七、如何在PyTorch中使用自动混合精度？
- 八、如何使用 AMP混合精度训练 ？
- ...

- [点击查看答案](https://articles.zsxq.com/id_0slrgoti6gvb.html)

### [图解分布式训练（六） —— Pytorch的 DeepSpeed 详细解析](https://articles.zsxq.com/id_2v6wv29ce8nn.html)

- 一、为什么需要 Deepspeed？
- 二、DeepSpeed 基本概念 介绍一下？
- 三、DeepSpeed 通信策略 介绍一下？
- 四、DeepSpeed 如何使用？
- 五、DeepSpeed 代码实现？
- 七、训练精度 介绍一下？
- 八、获取模型参数 介绍一下？
- ...

- [点击查看答案](https://articles.zsxq.com/id_2v6wv29ce8nn.html)

### [图解分布式训练（七）—— accelerate 分布式训练 详细解析](https://articles.zsxq.com/id_o5wkeionnqr7.html)

- 一、为什么需要 accelerate 分布式训练？
- 二、什么是 accelerate 分布式训练?
- 三、accelerate 分布式训练 原理讲解？
- 四、accelerate 分布式训练 如何实践？
- ...

- [点击查看答案](https://articles.zsxq.com/id_o5wkeionnqr7.html)

### [图解分布式训练（八）—— ZeRO 学习](https://articles.zsxq.com/id_600z63vou4nj.html)

- 一、什么是 3D 并行？
- 二、3D 并行 策略有哪些？
- 三、为什么需要 ZeRO？
- 四、ZeRO 的 核心思想是什么？
- 五、ZeRO 显存如何分配？
- 六、ZeRO 优化策略是怎么样？
- 七、ZeRO Offload后的计算流程是怎么样？
- ...

- [点击查看答案](https://articles.zsxq.com/id_600z63vou4nj.html)

## [大模型（LLMs）agent 面](https://articles.zsxq.com/id_9dfwi0ooio2z.html)

1. 如何给LLM注入领域知识？
2. 如果想要快速体验各种模型，该怎么办？
3. ...

- [点击查看答案](https://articles.zsxq.com/id_mzfogrjhkp17.html)

## [Token及模型参数准备篇](https://articles.zsxq.com/id_9oplu4014qx5.html)

1. 预训练数据 Token 重复 是否影响 模型性能？
2. SFT需要训练Token数？
3. ...

- [点击查看答案](https://articles.zsxq.com/id_9oplu4014qx5.html)

## [LLMs 位置编码篇](https://articles.zsxq.com/id_bmn80nar12c7.html)

- 1 什么是位置编码？
- 2 什么是绝对位置编码？
- 3 什么是相对位置编码？
- 4 旋转位置编码 RoPE篇
  - 4.1 旋转位置编码 RoPE 思路是什么？
  - 4.2 推导一下 旋转位置编码 RoPE ？
  - 4.3 旋转位置编码 RoPE 有什么优点？
  - 4.4 旋转位置编码 RoPE 被哪些 LLMs 应用？
- 5 长度外推问题篇
  - 5.1 什么是 长度外推问题？
  - 5.2 长度外推问题 的 解决方法 有哪些？
- 6 ALiBi (Attention with Linear Biases)篇
  - 6.1 ALiBi (Attention with Linear Biases) 思路是什么？
  - 6.2 ALiBi (Attention with Linear Biases) 的偏置矩阵是什么？有什么作用？
  - 6.3 ALiBi (Attention with Linear Biases) 有什么优点？
  - 6.4 ALiBi (Attention with Linear Biases)  被哪些 LLMs 应用？
- ...

- [点击查看答案](https://articles.zsxq.com/id_bmn80nar12c7.html)

## LLMs Tokenizer 篇

### [LLMs Tokenizer 篇](https://articles.zsxq.com/id_c1wrizv0im1a.html)

- Byte-Pair Encoding(BPE)篇
  - 1 Byte-Pair Encoding(BPE) 如何构建词典？
- WordPiece 篇
  - 1 WordPiece 与 BPE 异同点是什么？
- SentencePiece 篇
  - 简单介绍一下 SentencePiece 思路？
- 对比篇
  - 1 举例 介绍一下 不同 大模型LLMs 的分词方式？
  - 2 介绍一下 不同 大模型LLMs 的分词方式 的区别？
- ...

- [点击查看答案](https://articles.zsxq.com/id_c1wrizv0im1a.html)

### [怎么让英文大语言模型支持中文？（一） —— 构建中文tokenization](https://articles.zsxq.com/id_w0d2q29sueq7.html)

- 一、为什么需要 构建中文tokenization？
- 二、如何对 原始数据预处理？
- 三、如何构建中文的词库？
- 四、如何使用transformers库加载sentencepiece模型？
- 五、如何合并英文词表和中文词表？
- 六、怎么使用修改后的词表？
- 总结一下 构建中文tokenization？
- ...

- [点击查看答案](https://articles.zsxq.com/id_w0d2q29sueq7.html)

### [怎么让英文大语言模型支持中文？（二） —— 继续预训练篇](https://articles.zsxq.com/id_jprkwhrvf3tw.html)

- 一、为什么需要进行继续预训练？
- 二、如何对 继续预训练 数据预处理？
- 三、如何 构建模型？
- 四、如何 使用模型？
- ...

- [点击查看答案](https://articles.zsxq.com/id_jprkwhrvf3tw.html)

### [怎么让英文大语言模型支持中文？（三） —— 对预训练模型进行指令微调](https://articles.zsxq.com/id_p2wj7zadwxwb.html)

- 一、为什么需要对预训练模型进行指令微调？
- 二、对预训练模型进行指令微调 数据 如何处理？
- 三、对预训练模型进行指令微调 tokenization 如何构建？
- 四、对预训练模型进行指令微调 模型 如何构建？
- 五、是否可以结合 其他库 使用？
- ...

- [点击查看答案](https://articles.zsxq.com/id_p2wj7zadwxwb.html)

## [Layer normalization 篇](https://articles.zsxq.com/id_pzcgd4ovk098.html)

- Layer normalization-方法篇
  - Layer Norm 篇
    - Layer Norm 的计算公式写一下？
  - RMS Norm 篇 （均方根 Norm）
    - RMS Norm 的计算公式写一下？
    - RMS Norm 相比于 Layer Norm 有什么特点？
  - Deep Norm 篇
    - Deep Norm 思路？
    - 写一下 Deep Norm 代码实现？
  - Deep Norm 有什么优点？
- Layer normalization-位置篇
  - 1 LN 在 LLMs 中的不同位置 有什么区别么？如果有，能介绍一下区别么？
- Layer normalization 对比篇
  - LLMs 各模型分别用了 哪种 Layer normalization？
- ...

- [点击查看答案](https://articles.zsxq.com/id_pzcgd4ovk098.html)

## [LLMs 激活函数篇](https://articles.zsxq.com/id_6xm3wzzice2s.html)

- 1 介绍一下 FFN 块 计算公式？
- 2 介绍一下 GeLU 计算公式？
- 3 介绍一下 Swish 计算公式？
- 4 介绍一下 使用 GLU 线性门控单元的 FFN 块 计算公式？
- 5 介绍一下 使用 GeLU 的 GLU 块 计算公式？
- 6 介绍一下 使用 Swish 的 GLU 块 计算公式？
- 各LLMs 都使用哪种激活函数？
- ...

- [点击查看答案](https://articles.zsxq.com/id_6xm3wzzice2s.html)

## 大模型（LLMs）加速篇

### [大模型（LLMs）加速篇](https://articles.zsxq.com/id_w9wewc152eux.html)

- 1. 当前优化模型最主要技术手段有哪些？
- 2. 推理加速框架有哪一些？都有什么特点？
- 3 vLLM 篇
  - 3.1 vLLM 的 功能有哪些？
  - 3.2 vLLM 的 优点有哪些？
  - 3.3 vLLM 的 缺点有哪些？
  - 3.4 vLLM 离线批量推理？
  - 3.5 vLLM API Server？
- 4 Text generation inference 篇
  - 4.1 介绍一下 Text generation inference？
  - 4.2 Text generation inference 的 功能有哪些？
  - 4.3 Text generation inference 的 优点有哪些？
  - 4.4 Text generation inference 的 缺点有哪些？
  - 4.5 Text generation inference 的 使用docker运行web server？
- ...

- [点击查看答案](https://articles.zsxq.com/id_w9wewc152eux.html)

### [LLM（大语言模型）部署加速方法——PagedAttention篇](https://articles.zsxq.com/id_p22mjq881n3n.html)

- 一、vLLM 用于大模型并行推理加速 存在什么问题？
- 二、vLLM 如何 优化 大模型并行推理加速？
- 三、什么是 PagedAttention？
- 四、 PagedAttention 如何存储 连续的key和value？
- 五、 PagedAttention 技术细节？
- 六、 PagedAttention 如何 实现安全共享？
- 七、 PagedAttention 源码介绍？
- ...

- [点击查看答案](https://articles.zsxq.com/id_p22mjq881n3n.html)

### [大模型推理加速工具 —— vLLM](https://articles.zsxq.com/id_zw5h9ogvac2w.html)

- 一、引言
  - 1.1 前言
  - 1.2 为什么 需要 vLLM ?
  - 1.3 vLLM 具有哪些特点 ?
  - 1.4 vLLM 支持哪些 Huggingface 模型 ?
- 二、vLLM 性能如何？
- 三、vLLM 依赖包
- 四、vLLM 如何安装？
- 五、vLLM 如何使用？
- 六、vLLM 分布式推理与服务
- ...

- [点击查看答案](https://articles.zsxq.com/id_zw5h9ogvac2w.html)

### [LLM（大语言模型）部署加速方法——Faster Transformer篇](https://articles.zsxq.com/id_dd2gowztxtfg.html)

- 一、为什么需要 FasterTransformer？
- 二、FasterTransformer 介绍一下？
- 三、FasterTransformer 核心是什么？
- 四、FasterTransformer 优化？
- ...

- [点击查看答案](https://articles.zsxq.com/id_dd2gowztxtfg.html)

### [纯Python超轻量高性能LLM推理框架 —— LightLLM](https://articles.zsxq.com/id_9a643feq2b0b.html)

- 一、引言
  - 1.1 前言
  - 1.2 为什么 需要 LightLLM ?
  - 1.3 目前 LLM推理框架 有 哪些?
- 二、LightLLM 介绍一下？
  - 2.1 什么是 LightLLM ？
  - 2.2 Token Attention 介绍？
  - 2.3 Efficient Router 介绍？
- 三、LightLLM 性能表现 介绍？
- 四、LightLLM 依赖包 有哪些？
- 五、LightLLM  如何安装？
  - 5.1 下载 LightLLM
  - 5.2 安装 LightLLM 依赖
  - 5.3 安装 LightLLM
- 六、LightLLM 如何使用？
  - 6.1 启动 LightLLM 服务
- 填坑笔记
  - LightLLM 支持模型 LLMs 模型？
- ...

- [点击查看答案](https://articles.zsxq.com/id_9a643feq2b0b.html)

### [LLM推理技术之StreamingLLM：如何拥有无限长生成能力](hhttps://articles.zsxq.com/id_0ld3pfcmnhj6.html)

- 一、前言
  - 1.1 大型语言模型（LLM）存在什么问题？
  - 1.2 StreamingLLM 背景介绍
  - 1.3 StreamingLLM 核心问题？
  - 1.4 StreamingLLM 存在哪些挑战？
  - 1.5 目前主流地增加输入文本长度的方法有哪些？
- 二、StreamingLLM 的思路是什么？
- ...

- [点击查看答案](https://articles.zsxq.com/id_0ld3pfcmnhj6.html)

## [Attention 升级面](https://articles.zsxq.com/id_j0nwuo0frw2x.html)

- 1 传统 Attention 存在哪些问题？
- 2 Attention 优化方向
- 3 Attention 变体有哪些？
- 4 Multi-Query Attention 篇
  - 4.1 Multi-head Attention 存在什么问题？
  - 4.2 介绍一下 Multi-Query Attention？
  - 4.3 对比一下 Multi-head Attention 和 Multi-Query Attention？
  - 4.4 Multi-Query Attention 这样做的好处是什么？
  - 4.5 有 哪些模型 是 使用 Multi-Query Attention？
- 5 Grouped-query Attention
  - 5.1 什么是 Grouped-query Attention？
  - 5.2 有哪些大模型使用 Grouped-query Attention？
- 6 FlashAttention 介绍一下
- 7 并行 transformer block 介绍一下？
- ...

- [点击查看答案](https://articles.zsxq.com/id_j0nwuo0frw2x.html)

## 大模型幻觉（LLM Hallucination）面

### [大模型幻觉（LLM Hallucination）面](https://articles.zsxq.com/id_schwrdmvmhr7.html)

- 一、什么是大模型幻觉？
- 二、为什么LLM会产生幻觉？
- 三、为什么需要解决LLM的幻觉问题？
- 四、幻觉一定是有害的吗？
- 五、幻觉有哪些不同类型？
- 六、如何度量幻觉？
- 七、如何缓解LLM幻觉？
  - 7.1 通过使用外部知识验证主动检测和减轻幻觉
  - 7.2 事实核心采样
  - 7.3 SelfCheckGPT
- 八、LLMs什么时候最容易产生幻觉？
- ...

- [点击查看答案](https://articles.zsxq.com/id_schwrdmvmhr7.html)

### [大模型的幻觉问题篇](https://articles.zsxq.com/id_8mr4mlhe5q1x.html)

- 一、什么是 大模型幻觉问题？
- 二、为什么 会 出现 大模型幻觉问题？
- 三、如何 评估 大模型幻觉问题？
- 四、如何 缓解 大模型幻觉问题？
- ...

- [点击查看答案](https://articles.zsxq.com/id_8mr4mlhe5q1x.html)

### [大模型的幻觉问题篇](https://articles.zsxq.com/id_tbezgzifowzp.html)

- 一、为什么 会 出现 大模型幻觉？
- 二、如何 缓解 大模型幻觉？
- ...

- [点击查看答案](https://articles.zsxq.com/id_tbezgzifowzp.html)

## LLMs 对比篇

### [LLMs 对比篇](https://articles.zsxq.com/id_tbezgzifowzp.html)

- LLMs 训练数据 和 数据量 对比如何？
- ...

- [点击查看答案](https://articles.zsxq.com/id_tbezgzifowzp.html)

### [百川智能baichuan7B、13B、53B、baichuan2 总结篇](https://articles.zsxq.com/id_ma6pw7v2g9pi.html)

- 一、baichuan-7B篇
  - 1. 你了解baichuan-7B解构么？介绍一下？
  - 2. baichuan-7B 如何 收集原始数据并 构建 训练数据？
  - 3. baichuan-7B 如何 提高 训练稳定性和吞吐？
- 二、baichuan-13B篇
  - 1. 相比于 baichuan-7B，baichuan-13B 的 特点体现在哪里？
  - 2. 如何 对 baichuan-13B 进行推理和部署？
  - 3. 如何 对 baichuan-13B 进行微调？
- 三、baichuan-53B篇
  - 3.1 baichuan-53B 相比于 baichuan-7B 和 baichuan-13B 有哪些优势？
  - 3.2 baichuan-53B 如何对 预训练数据 做处理？
  - 3.3 baichuan-53B 如何进行 搜索增强？
- 四、baichuan2篇
  - 4.1 baichuan2 与 其他大模型 对比
- 五、baichuan 数据构建篇
  - 5.1 baichuan 进行微调时，领域数据：通用数据配比？
- ...

- [点击查看答案](https://articles.zsxq.com/id_ma6pw7v2g9pi.html)

## 思维链 Chain-of-Thought（COT）篇

### [思维链 Chain-of-Thought（COT）篇](https://articles.zsxq.com/id_1cjjxf95az70.html)

- 一、什么是思维链提示？
- 二、思维链提示本质是什么？
- 三、思维链提示 与 标准的提示学习方法有什么不同?
- 四、思维链提示 为什么可以提高语言模型的复杂推理能力?它的优势在哪里?
- 五、思维链提示 适用场景 有 哪些？
- 六、思维链提示 目前还存在哪些不足点？
- 七、思维链提示 对推动语言模型复杂推理能力研究有哪些启发和影响?
- 八、思维链提示 对实现真正的通用人工智能仍面临哪些挑战?
- 九、如何通过增加模型规模来获得语言模型强大的思路链推理能力的?这与模型获得的哪些能力有关?
- 十、你认为可以在哪些其他方面应用“思路链提示”这一思路来提升语言模型的能力?
- 十一、如果需要你对 思维链提示 进行改进，你觉得你会改进哪些地方？
- 十二、思维链提示 未来研究方向？
- ...

- [点击查看答案](https://articles.zsxq.com/id_1cjjxf95az70.html)

### [思维链 Chain-of-Thought（COT）变体篇](https://articles.zsxq.com/id_sw5aljfzswiv.html)

- 思维链 Chain-of-Thought（COT）：思维链的启蒙
  - 1. 什么是 思维链 Chain-of-Thought（COT）？
  - 2. 思维链 Chain-of-Thought（COT）是思路是什么？
  - 3. 思维链 Chain-of-Thought（COT）存在问题？
- 思维树 Tree of Thoughts（TOT）：一种用树结构解决复杂问题的方法
  - 1. 为什么需要 思维树 Tree of Thoughts（TOT）？
  - 2. 什么是 思维树 Tree of Thoughts（TOT）？
  - 3. 思维树 Tree of Thoughts（TOT）涉及问题有哪些？
- 思维图 Graph of Thoughts（GOT）：一种把思维链过程建模层图结构的方法
  - 1. 为什么 需要 思维图 Graph of Thoughts（GOT）？
  - 2. 什么是 思维图 Graph of Thoughts（GOT） ？
  - 3. 思维图 Graph of Thoughts（GOT）核心思想是什么 ？
- 思维算法 Algorithm of Thoughts（AOT）：一种用DFS/BFS示例解决问题的方法
  - 1. 为什么 需要 思维算法 Algorithm of Thoughts（AOT）？
  - 2. 思维算法 Algorithm of Thoughts（AOT）思路是什么？
  - 3. 思维算法 Algorithm of Thoughts（AOT） vs 其他 COT 的 区别？
- 思维链 Chain-of-Thought（COT） 有哪些 应用场景？
- 思维链 Chain-of-Thought（COT） 有哪些 局限性？
- ...

- [点击查看答案](https://articles.zsxq.com/id_sw5aljfzswiv.html)

## [Graph RAG（Retrieval-Augmented Generation） 面 —— 一种 基于知识图谱的大模型检索增强实现策略](hhttps://articles.zsxq.com/id_dwhonmw976n7.html)

- 一、为什么需要 Graph RAG？
- 二、什么是 Graph RAG？
- 三、Graph RAG 思路介绍？
- 四、用代码 介绍 Graph RAG ？
- 五、用 示例 介绍 Graph RAG ？
- 六、Graph RAG 排序优化方式？
- ...

- [点击查看答案](https://articles.zsxq.com/id_dwhonmw976n7.html)

## [大模型生成去重技术面](https://articles.zsxq.com/id_rvwtlip4gi5e.html)

- 一、什么是生成式大模型？
- 二、大模型是怎么让生成的文本丰富而不单调的呢？
- 三、生成式大模型 存在哪些问题？
- 四、生成式大模型 为什么会出现 重复生成现象？
- 五、生成式大模型 有哪些解决方法？
  - 5.1 Unlikelihood Training
  - 5.2 Repetition Penalty
  - 5.3 Contrastive Search
  - 5.4 Beam Search
  - 5.5 TopK sampling
  - 5.6 Nucleus sampler
  - 5.7 Temperature
  - 5.8 No repeat ngram size
  - 5.9 重复率指标检测
- ...

- [点击查看答案](https://articles.zsxq.com/id_rvwtlip4gi5e.html)


