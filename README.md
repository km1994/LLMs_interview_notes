# LLMs 千面郎君

> 介绍：本项目是作者们根据个人面试和经验总结出的 大模型(LLMs)面试准备的学习笔记与资料，该资料目前包含 大模型(LLMs)各领域的 面试题积累。

<img src="img/微信截图_20230918094559.png" width="50%" >
> LLMs 千面郎君 面试交流群 (注：人满 可 添加 小编wx：yzyykm666 加群！)

<img src="img/微信截图_20210301212242.png" width="50%" >

## [大模型（LLMs）基础面](https://articles.zsxq.com/id_zpsd43wksbp2.html)

1. 目前 主流的开源模型体系 有哪些？
2. prefix LM 和 causal LM 区别是什么？
3. 涌现能力是啥原因？
4. 大模型LLM的架构介绍？

- [点击查看答案](https://articles.zsxq.com/id_zpsd43wksbp2.html)

## [大模型（LLMs）进阶面](https://articles.zsxq.com/id_i5m3wfkdzwq9.html)

1. LLMs 复读机问题
   1. 什么是 LLMs 复读机问题？
   2. 为什么会出现 LLMs 复读机问题？
   3. 如何缓解 LLMs 复读机问题？
2. llama 系列问题
   1. llama 输入句子长度理论上可以无限长吗？
3. 什么情况用Bert模型，什么情况用LLaMA、ChatGLM类大模型，咋选？
4. 各个专业领域是否需要各自的大模型来服务？
5. 如何让大模型处理更长的文本？

- [点击查看答案](https://articles.zsxq.com/id_i5m3wfkdzwq9.html)

## [大模型（LLMs）微调面](https://articles.zsxq.com/id_u62mcnga3jkd.html)

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

- [点击查看答案](https://articles.zsxq.com/id_u62mcnga3jkd.html)

## 大模型（LLMs）langchain 面

### [大模型（LLMs）langchain 面](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

1. 什么是 LangChain?
2. LangChain 包含哪些 核心概念？
   1. LangChain 中 Components and Chains 是什么？
   2. LangChain 中 Prompt Templates and Values 是什么？
   3. LangChain 中 Example Selectors 是什么？
   4. LangChain 中 Output Parsers 是什么？
   5. LangChain 中 Indexes and Retrievers 是什么？
   6. LangChain 中  Chat Message History 是什么？
   7. LangChain 中  Agents and Toolkits 是什么？
3. 什么是 LangChain Agent?
4. 如何使用 LangChain ?
5. LangChain 支持哪些功能?
6. 什么是 LangChain model?
7. LangChain 包含哪些特点?
8. LangChain 如何使用?
   1. LangChain 如何调用 LLMs 生成回复？
   2. LangChain 如何修改 提示模板？
   3. LangChain 如何链接多个组件处理一个特定的下游任务？
   4. LangChain 如何Embedding \& vector store？
9. LangChain 存在哪些问题及方法方案？
   1.  LangChain 低效的令牌使用问题
   2.  LangChain 文档的问题
   3.  LangChain 太多概念容易混淆，过多的“辅助”函数问题
   4.  LangChain 行为不一致并且隐藏细节问题
   5.  LangChain 缺乏标准的可互操作数据类型问题
10. LangChain 替代方案？

### [基于LLM+向量库的文档对话 经验面](https://articles.zsxq.com/id_dfwoe4vgpang.html)

1. 基于LLM+向量库的文档对话 基础面
   1. LLMs 存在模型幻觉问题，请问如何处理？
   2. 基于LLM+向量库的文档对话 思路是怎么样？
   3. 基于LLM+向量库的文档对话 核心技术是什么？
   4. 基于LLM+向量库的文档对话 prompt 模板 如何构建？
2. 基于LLM+向量库的文档对话 优化面
   1. 痛点1：文档切分粒度不好把控，既担心噪声太多又担心语义信息丢失
   2. 痛点2：在基于垂直领域 表现不佳
   3. 痛点3：langchain 内置 问答分句效果不佳问题
   4. 痛点4：如何 尽可能召回与query相关的Document 问题
   5. 痛点5：如何让LLM基于query和context得到高质量的response
3. 基于LLM+向量库的文档对话 工程示例面
   1. 本地知识库问答系统（Langchain-chatGLM）
      1. 避坑记录

## [大模型（LLMs）参数高效微调(PEFT) 面](https://articles.zsxq.com/id_ahk2br3igwx9.html)

1. 微调方法是啥？如何微调？
2. LoRA 篇
   1. LoRA权重是否可以合入原模型？
   2. ChatGLM-6B LoRA后的权重多大？
   3. LoRA 微调 特点
   4. LoRA微调方法为啥能加速训练？
   5. 如何在已有LoRA模型上继续训练？
3. 对比篇
   1. 微调方法批处理大小模式GPU显存速度
   2. Peft 和 全量微调区别？

- [点击查看答案](https://articles.zsxq.com/id_ahk2br3igwx9.html)

## [大模型（LLMs）推理面](https://articles.zsxq.com/id_64vc5vvwpobv.html)

1. 为什么大模型推理时显存涨的那么多还一直占着？
2. 大模型在gpu和cpu上推理速度如何？
3. 推理速度上，int8和fp16比起来怎么样？
4. 大模型有推理能力吗？
5. 大模型生成时的参数怎么设置？
6. 有哪些省内存的大语言模型训练/微调/推理方法？
7. 如何让大模型输出合规化
8. 应用模式变更

- [点击查看答案](https://articles.zsxq.com/id_64vc5vvwpobv.html)

## [大模型（LLMs）评测面](https://articles.zsxq.com/id_z3bis84sxb9x.html)

1. 大模型怎么评测？
2. 大模型的honest原则是如何实现的？模型如何判断回答的知识是训练过的已知的知识，怎么训练这种能力？

- [点击查看答案](https://articles.zsxq.com/id_z3bis84sxb9x.html)

## [大模型（LLMs）强化学习面](https://articles.zsxq.com/id_uru2bfwhg34c.html)

1. 奖励模型需要和基础模型一致吗？
2. RLHF 在实践过程中存在哪些不足？
3. 如何解决 人工产生的偏好数据集成本较高，很难量产问题？
4. 如何解决三个阶段的训练（SFT->RM->PPO）过程较长，更新迭代较慢问题？
5. 如何解决 PPO 的训练过程同时存在4个模型（2训练，2推理），对计算资源的要求较高 问题？

- [点击查看答案](https://articles.zsxq.com/id_uru2bfwhg34c.html)

## [大模型（LLMs）软硬件配置面](https://articles.zsxq.com/id_m5q8zk3wo84k.html)

1. 建议的软件环境是什么？

- [点击查看答案](https://articles.zsxq.com/id_m5q8zk3wo84k.html)

## [大模型（LLMs）训练集面](https://articles.zsxq.com/id_jwvpaujrojtt.html)

1. SFT（有监督微调）的数据集格式？
2. RM（奖励模型）的数据格式？
3. PPO（强化学习）的数据格式？
4. 找数据集哪里找？
5. 微调需要多少条数据？
6. 有哪些大模型的训练集？
7. 进行领域大模型预训练应用哪些数据集比较好？

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

- [点击查看答案](https://articles.zsxq.com/id_jhiocx89p3su.html)

## [大模型（LLMs）分布式训练面](https://articles.zsxq.com/id_lk1wnxtwnr9a.html)

- 理论篇
  - 想要训练1个LLM，如果只想用1张显卡，那么对显卡的要求是什么？
  - 如果有N张显存足够大的显卡，怎么加速训练？
  - 如果显卡的显存不够装下一个完整的模型呢？
  - PP推理时，是一个串行的过程，1个GPU计算，其他空闲，有没有其他方式？
  - 3种并行方式可以叠加吗？
  - Colossal-AI 有1D/2D/2.5D/3D，是什么情况？
  - 除了3D并行有没有其他方式大规模训练？
  - 有了ZeRO系列，为什么还需要3D并行？
  - 平民适不适合玩3D并行？
  - 平民适不适合直接上多机多卡的ZeRO3（万兆网）？
- 实践篇
  - 假如有超多的8卡A100节点（DGX A100），如何应用3D并行策略？
  - 如果想构这样一个大规模并行训练系统，训练框架如何选？
  - 训练框架如何选？
- 并行化策略选择篇
- 问题篇
  - 推理速度验证
  - 并行化训练加速
  - deepspeed 训练过程，报找不主机
  - 为什么 多机训练效率不如单机？
  - 多机训练不通，DeepSPeed配置问题

- [点击查看答案](https://articles.zsxq.com/id_lk1wnxtwnr9a.html)

## [大模型（LLMs）agent 面](https://articles.zsxq.com/id_mzfogrjhkp17.html)

1. 如何给LLM注入领域知识？
2. 如果想要快速体验各种模型，该怎么办？

- [点击查看答案](https://articles.zsxq.com/id_mzfogrjhkp17.html)
