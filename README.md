# LLMs 千面郎君

> 介绍：本项目是作者们根据个人面试和经验总结出的 大模型(LLMs)面试准备的学习笔记与资料，该资料目前包含 大模型(LLMs)各领域的 面试题积累。

<img src="img/微信截图_20230918094559.png" width="50%" >
> LLMs 千面郎君 面试交流群 (注：人满 可 添加 小编wx：yzyykm666 加群！)

<img src="img/微信截图_20210301212242.png" width="50%" >


## 一、大模型（LLMs）基础面 

### [大模型（LLMs）基础面](https://articles.zsxq.com/id_mw52p1pfbzql.html) 

- 1 目前 主流的开源模型体系 有哪些？
- 2 prefix Decoder 和 causal Decoder 和 Encoder-Decoder 区别是什么？
- 3 大模型LLM的 训练目标 是什么？
- 4 涌现能力是啥原因？
- 5 为何现在的大模型大部分是Decoder only结构？
- 6 简单 介绍一下 大模型【LLMs】？
- 7 大模型【LLMs】后面跟的 175B、60B、540B等 指什么？
- 8 大模型【LLMs】具有什么优点？
- 9 大模型【LLMs】具有什么缺点？
- 10 encoder-only, decoder-only, encoder-decoder的区别?
- 11 BART、llama、gpt、t5、palm等主流模型异同点?
- 12 prefix LM 和 causal LM 区别是什么?

- [点击查看答案](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### [Layer normalization 篇](https://articles.zsxq.com/id_pzcgd4ovk098.html)

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

- [点击查看答案](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### [LLMs 激活函数篇](https://articles.zsxq.com/id_6xm3wzzice2s.html) 

- 1 介绍一下 FFN 块 计算公式？
- 2 介绍一下 GeLU 计算公式？
- 3 介绍一下 Swish 计算公式？
- 4 介绍一下 使用 GLU 线性门控单元的 FFN 块 计算公式？
- 5 介绍一下 使用 GeLU 的 GLU 块 计算公式？
- 6 介绍一下 使用 Swish 的 GLU 块 计算公式？
- 7 各LLMs 都使用哪种激活函数？
- 8 Adam优化器和SGD的区别？

- [点击查看答案](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### [Attention 升级面](https://articles.zsxq.com/id_u67us9zex93d.html) 

- [Attention 升级面](https://articles.zsxq.com/id_u67us9zex93d.html) 
  - 1 传统 Attention 存在哪些问题？
  - 2 Attention 有哪些 优化方向？
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
  - 6 FlashAttention
    - 6.1 为什么需要  FlashAttention？
    - 6.2 简单介绍一下 FlashAttention？
    - 6.3 简单介绍一下 FlashAttention 核心？
    - 6.4 介绍一下 FlashAttention 优点？
    - 6.5 介绍一下 FlashAttention 代表模型？
  - 7 并行 transformer block
  - 8 attention计算复杂度以及如何改进？
  - 9 Paged Attention篇
    - 9.1 简单介绍一下 Paged Attention？
  - 对比篇
    - 1、MHA，GQA，MQA 三种注意力机制是否了解?区别是什么?

- [点击查看答案](https://articles.zsxq.com/id_u67us9zex93d.html)

- [跨注意力机制（Cross-Attention）篇](https://articles.zsxq.com/id_gwn416686pic.html) 
  - 一、为什么需要 跨注意力机制（Cross-Attention）？
  - 二、介绍一些 跨注意力机制（Cross-Attention）？
  - 三、Cross Attention 和 Self Attention 篇
    - 3.1 Cross Attention 和 Self Attention 都是基于注意力机制的，有什么相同点？
    - 3.2 Cross Attention 和 Self Attention 都是基于注意力机制的，有什么不同点？
  - 四、Cross Attention 和 多头注意力（Multi-Head Attention）篇
    - 4.2 Cross Attention 和 多头注意力（Multi-Head Attention） 都是基于注意力机制的，有什么异同点？
  - 五、Cross Attention 代码实现
  - 六、Cross Attention 应用场景
  - 七、Cross Attention 的优势和挑战？

- [点击查看答案](https://articles.zsxq.com/id_gwn416686pic.html)

### [transformers 操作篇](https://articles.zsxq.com/id_rsll7gsd8va5.html) 

- 1. 如何 利用 transformers 加载 Bert 模型？
- 2. 如何 利用 transformers 输出 Bert 指定 hidden\_state？
- 3. BERT 获取最后一层或每一层网络的向量输出

- [点击查看答案](https://articles.zsxq.com/id_rsll7gsd8va5.html)

### [LLMs 损失函数篇](https://articles.zsxq.com/id_q0ajjlbc8493.html) 

- 一、介绍一下 KL 散度？
- 二、交叉熵损失函数写一下，物理意义是什么？
- 三、KL 散度与交叉熵的区别？
- 四、多任务学习各loss差异过大怎样处理？
- 五、分类问题为什么用交叉熵损失函数不用均方误差（MSE）？
- 六、什么是信息增益？
- 七、多分类的分类损失函数(Softmax)？
- 八、softmax和交叉熵损失怎么计算，二值交叉熵呢？
- 九、如果softmax的e次方超过float的值了怎么办？

- [点击查看答案](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### [相似度函数篇](https://articles.zsxq.com/id_wp25j5xr8ocw.html) 

- 一、除了cosin还有哪些算相似度的方法
- 二、了解对比学习嘛？
- 三、对比学习负样本是否重要？负样本构造成本过高应该怎么解决？

- [点击查看答案](https://articles.zsxq.com/id_wp25j5xr8ocw.html)

## [二、大模型（LLMs）进阶面](https://articles.zsxq.com/id_xr65bxpcsnoh.html) 

- 一、什么是生成式大模型？
- 二、大模型是怎么让生成的文本丰富而不单调的呢？
- 三、LLMs 复读机问题
  - 3.1 什么是 LLMs 复读机问题？
  - 3.2 为什么会出现 LLMs 复读机问题？
  - 3.3 如何缓解 LLMs 复读机问题？
- 四、llama 系列问题
  - 4.1 llama 输入句子长度理论上可以无限长吗？
- 五、什么情况用Bert模型，什么情况用LLaMA、ChatGLM类大模型，咋选？
- 六、各个专业领域是否需要各自的大模型来服务？
- 七、如何让大模型处理更长的文本？

- [点击查看答案](https://articles.zsxq.com/id_xr65bxpcsnoh.html)

## 三、大模型（LLMs）微调面

### [大模型（LLMs）微调面](https://articles.zsxq.com/id_kv7jdah2zw5n.html) 

- 39 大模型 sft 过程中，为什么会出现第二个epoch的时候loss会突然下降问题？
- 1 如果想要在某个模型基础上做全参数微调，究竟需要多少显存？
- 2 为什么SFT之后感觉LLM傻了?
- 3 SFT 指令微调数据 如何构建?
    - 3.1 提升sft的prompt的代表性有什么好的方法？
    - 3.2 提升sft的prompt的数据量有什么好的方法？
- 4 领域模型Continue PreTrain 数据选取？
- 5 领域数据训练后，通用能力往往会有所下降，如何缓解模型遗忘通用能力？
- 6 领域模型Continue PreTrain ，如何 让模型在预训练过程中就学习到更多的知识？
- 7 进行SFT操作的时候，基座模型选用Chat还是Base?
- 8 领域模型微调 指令\&数据输入格式 要求？
- 9 领域模型微调 领域评测集 构建？
- 10 领域模型词表扩增是不是有必要的？
- 11 如何训练自己的大模型？
- 12 训练中文大模型有啥经验？
- 13 指令微调的好处？
- 14 预训练和微调哪个阶段注入知识的？
- 15 想让模型学习某个领域或行业的知识，是应该预训练还是应该微调？
- ...

- [点击查看答案](https://articles.zsxq.com/id_kv7jdah2zw5n.html)

### [大模型 SFT Trick 篇](https://articles.zsxq.com/id_srd92pvnjwmu.html)

- 一、常见 SFT的开发流程是如何的？
- 二、训练数据要注重什么？
- 三、大 size 和小 size 模型的选择？
- 四、多任务训练时怎么确保每个任务都优秀？
- 五、SFT真的不能学到知识？
- 六、怎么科学挑选数据集？
- ...

- [点击查看答案](https://articles.zsxq.com/id_srd92pvnjwmu.html)

### [大模型（LLMs）训练经验帖](https://articles.zsxq.com/id_06n25d9wjs0e.html)

- 分布式训练框架选择？
- LLMs 训练时 有哪些有用的建议？
- 模型大小如何选择？
- 加速卡如何选择？

- [点击查看答案](https://articles.zsxq.com/id_06n25d9wjs0e.html)

## 四、大模型（LLMs）langchain 面

### [大模型（LLMs）langchain 面](https://articles.zsxq.com/id_ve2dgaiqrjzv.html) 

- 一、什么是 LangChain?
- 二、LangChain 包含哪些 核心概念？
  - 2.1 LangChain 中 Components and Chains 是什么？
  - 2.2 LangChain 中 Prompt Templates and Values 是什么？
  - 2.3 LangChain 中 Example Selectors 是什么？
  - 2.4 LangChain 中 Output Parsers 是什么？
  - 2.5 LangChain 中 Indexes and Retrievers 是什么？
  - 2.6 LangChain 中  Chat Message History 是什么？
  - 2.7 LangChain 中  Agents and Toolkits 是什么？
- ...

- [点击查看答案](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### [多轮对话中让AI保持长期记忆的8种优化方式篇](https://articles.zsxq.com/id_3qgicwcwzjpi.html) 

- 一、前言
- 二、Agent 如何获取上下文对话信息？
  - 2.1 获取全量历史对话
  - 2.2 滑动窗口获取最近部分对话内容
  - ...

- [点击查看答案](https://articles.zsxq.com/id_3qgicwcwzjpi.html)

### [基于langchain RAG问答应用实战](https://articles.zsxq.com/id_3kw7snrk2rql.html) 

- [点击查看答案](https://articles.zsxq.com/id_3kw7snrk2rql.html)

## 五、大模型（LLMs）RAG 检索增强生成面 

### 5.1 大模型（LLMs）RAG 入门篇

#### [基于LLM+向量库的文档对话 经验面](https://articles.zsxq.com/id_xk58m8ok2sob.html)

- 一、基于LLM+向量库的文档对话 基础面
  - 1.1 为什么 大模型 需要 外挂(向量)知识库？
  - 1.2. 基于LLM+向量库的文档对话 思路是怎么样？
  - 1.3. 基于LLM+向量库的文档对话 核心技术是什么？
  - 1.4. 基于LLM+向量库的文档对话 prompt 模板 如何构建？
- ...

- [点击查看答案](https://articles.zsxq.com/id_xk58m8ok2sob.html)

#### [RAG（Retrieval-Augmented Generation）面](https://articles.zsxq.com/id_xk58m8ok2sob.html) 

- 一、LLMs 已经具备了较强能力了，存在哪些不足点?
- 二、什么是 RAG?
  - 2.1 R：检索器模块
    - 2.1.1 如何获得准确的语义表示？
    - 2.1.2 如何协调查询和文档的语义空间？
    - 2.1.3 如何对齐检索模型的输出和大语言模型的偏好？
  - 2.2 G：生成器模块
    - 2.2.1 生成器介绍
    - 2.2.2 如何通过后检索处理提升检索结果？
    - 2.2.3 如何优化生成器应对输入数据？
- 三、使用 RAG 的好处?
- ...

- [点击查看答案](https://articles.zsxq.com/id_xk58m8ok2sob.html)

#### 5.2 大模型（LLMs）RAG 版面分析篇

### [大模型（LLMs）RAG —— pdf解析关键问题](https://articles.zsxq.com/id_2693k55it84w.html)

- 一、为什么需要进行pdf解析？
- 二、为什么需要 对 pdf 进行解析？
- 三、pdf解析 有哪些方法，对应的区别是什么？
- 四、pdf解析 存在哪些问题？
- ...

- [点击查看答案](https://articles.zsxq.com/id_2693k55it84w.html)

### [大模型（LLMs）RAG 版面分析——表格识别方法篇](https://articles.zsxq.com/id_7x4qv94hxv8r.html)

- 一、为什么需要识别表格？
- 二、介绍一下 表格识别 任务？
- 三、有哪些 表格识别方法？
  - 3.1 传统方法
  - 3.2 pdfplumber表格抽取
    - 3.2.1 pdfplumber 如何进行 表格抽取？
    - 3.2.2 pdfplumber 常见的表格抽取模式？
  - ...

- [点击查看答案](https://articles.zsxq.com/id_7x4qv94hxv8r.html)

### [大模型（LLMs）RAG 版面分析——文本分块面](https://articles.zsxq.com/id_iw7debl8akxh.html)

- 一、为什么需要对文本分块？
- 二、能不能介绍一下常见的文本分块方法？
  - 2.1 一般的文本分块方法
  - 2.2 正则拆分的文本分块方法
  - 2.3 Spacy Text Splitter 方法
  - 2.4 基于 langchain 的 CharacterTextSplitter 方法
  - ...

- [点击查看答案](https://articles.zsxq.com/id_iw7debl8akxh.html)

### 5.3 大模型（LLMs）RAG 检索策略篇

#### [大模型外挂知识库优化——如何利用大模型辅助召回？](https://articles.zsxq.com/id_oznm6qixjw61.html)

- 一、为什么需要使用大模型辅助召回？
  - 策略一： HYDE
    - 1. 介绍一下 HYDE 思路？
    - 2. 介绍一下 HYDE 问题？
  - 策略二： FLARE
    - 1. 为什么 需要 FLARE ？
    - 2. FLARE 有哪些召回策略？

- [点击查看答案](https://articles.zsxq.com/id_oznm6qixjw61.html)

#### [大模型外挂知识库优化——负样本样本挖掘篇](https://articles.zsxq.com/id_wa7nl8wsuilh.html)

- 一、为什么需要构建负难样本？
- 二、负难样本构建方法篇
  - 2.1 随机采样策略（Random Sampling）方法
  - 2.2 Top-K负例采样策略（Top-K Hard Negative Sampling）方法
  - ...

- [点击查看答案](https://articles.zsxq.com/id_wa7nl8wsuilh.html)

### 5.4 大模型（LLMs）RAG 评测篇

#### [RAG（Retrieval-Augmented Generation）评测面](https://articles.zsxq.com/id_vjwt6uzml13l.html)

- 一、为什么需要 对 RAG 进行评测？
- 二、RAG 有哪些评估方法？
- 三、RAG 有哪些关键指标和能力？
- 四、RAG 有哪些评估框架？

- [点击查看答案](https://articles.zsxq.com/id_vjwt6uzml13l.html)

### 5.5 大模型（LLMs）RAG 优化策略篇

#### [检索增强生成(RAG) 优化策略篇](https://articles.zsxq.com/id_gu4p7gszsh82.html)

- 一、RAG基础功能篇
  - 1.1 RAG 工作流程
- 二、RAG 各模块有哪些优化策略？
- 三、RAG 架构优化有哪些优化策略？
  - 3.1 如何利用 知识图谱（KG）进行上下文增强？
    - 3.1.1 典型RAG架构中，向量数据库进行上下文增强 存在哪些问题？
    - 3.1.2 如何利用 知识图谱（KG）进行上下文增强？
  - ...

- [点击查看答案](https://articles.zsxq.com/id_gu4p7gszsh82.html)

#### [RAG 关键痛点及对应解决方案](https://articles.zsxq.com/id_1bmbedojsj0t.html)

- 前言
- 问题一：内容缺失问题
  - 1.1 介绍一下 内容缺失问题？
  - 1.2 如何 解决 内容缺失问题？
- 问题二：错过排名靠前的文档
  - 2.1 介绍一下 错过排名靠前的文档 问题？
  - 2.2 如何 解决 错过排名靠前的文档 问题？
- 问题三：脱离上下文 — 整合策略的限制
  - 3.1 介绍一下 脱离上下文 — 整合策略的限制 问题？
  - 3.2 如何 解决 脱离上下文 — 整合策略的限制 问题？
- 问题四：未能提取答案
  - 4.1 介绍一下 未能提取答案 问题？
  - 4.2 如何 解决 未能提取答案 问题？
- ...

- [点击查看答案](https://articles.zsxq.com/id_1bmbedojsj0t.html)

#### [大模型（LLMs）RAG 优化策略 —— RAG-Fusion篇](https://articles.zsxq.com/id_4ce04xwvic1z.html)

- 一、RAG 有哪些优点？
- 二、RAG 存在哪些局限性？
- 三、为什么 需要 RAG-Fusion？
- 四、说一下 RAG-Fusion 核心技术？
- 五、说一下 RAG-Fusion 工作流程？
  - ...

- [点击查看答案](https://articles.zsxq.com/id_4ce04xwvic1z.html)

### 5.6 大模型（LLMs）Graph RAG篇

#### [Graph RAG（Retrieval-Augmented Generation） 面 —— 一种 基于知识图谱的大模型检索增强实现策略](https://articles.zsxq.com/id_dwhonmw976n7.html)

- 一、为什么需要 Graph RAG？
- 二、什么是 Graph RAG？
- 三、Graph RAG 思路介绍？
- 四、用代码 介绍 Graph RAG ？
- 五、用 示例 介绍 Graph RAG ？
- 六、Graph RAG 排序优化方式？

- [点击查看答案](https://articles.zsxq.com/id_dwhonmw976n7.html)

## 六、大模型（LLMs）参数高效微调(PEFT) 面

### [大模型（LLMs）参数高效微调(PEFT) 面](https://articles.zsxq.com/id_ipkod91a939n.html)

- 1. 微调方法是啥？如何微调？
- 2. 为什么需要 PEFT？
- 3. 介绍一下 PEFT？
- 4. PEFT 有什么优点？
- ...

- [点击查看答案](https://articles.zsxq.com/id_ipkod91a939n.html)

### [配器微调（Adapter-tuning）篇](https://articles.zsxq.com/id_0n6pfw0wz3xb.html)

- 一、为什么 需要 适配器微调（Adapter-tuning）？
- 二、适配器微调（Adapter-tuning）思路？
- 三、 适配器微调（Adapter-tuning）特点是什么？
- 四、AdapterFusion 思路 是什么？
- ...
- [点击查看答案](https://articles.zsxq.com/id_0n6pfw0wz3xb.html)

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
  - ...

- [点击查看答案](https://articles.zsxq.com/id_662wpbw47gtj.html)

### [LoRA 系列篇](https://articles.zsxq.com/id_gjkhd8xn4pvt.html) 

一、LoRA篇
    - 1.1 什么是 LoRA？
    - 1.2 LoRA 的思路是什么？
    - 1.3 LoRA 的特点是什么？
    - 1.4 简单描述一下 LoRA?
    - 1.5 解释一下 LORA 微调的原理和计算流程？
- 二、LoRA变体篇
    - 2.1 QLoRA篇
        - 2.1.1 QLoRA 的思路是怎么样的？
        - 2.1.2 QLoRA 的特点是什么？
        - 2.1.3 QLORA相比LORA做了哪些改进?
    - 2.2 AdaLoRA篇
    -   .2.1 AdaLoRA 的思路是怎么样的？
    - 2.3 LongLoRA篇
        - 2.3.1 为什么需要 LongLoRA？
        - 2.3.2 LongLoRA 思路是什么？
        - 2.3.3 介绍一下 shift short attention？
- 三、Lora的矩阵怎么初始化？为什么要初始化为全0？
- ...

- [点击查看答案](https://articles.zsxq.com/id_gjkhd8xn4pvt.html)

### [如何使用 PEFT库 中 LoRA？](https://articles.zsxq.com/id_8lx1t1t3w4qf.html) 

- 一、前言
- 二、如何 配置 LoraConfig？
- 三、模型 加入PEFT策略
  - 3.1 模型加载 策略有哪些？
  - 3.2 模型显存占用的部分有哪些？
  - 3.3 模型显存占用 优化策略？
    - 3.3.1 8bit量化 优化策略？
    - 3.3.2 梯度检查 优化策略？
  - 3.4 如何 向 模型 加入PEFT策略？
- ...

- [点击查看答案](https://articles.zsxq.com/id_8lx1t1t3w4qf.html)

### [大模型 SFT 方式对比篇](https://articles.zsxq.com/id_e2piver2uzei.html) 

- 一、SFT 微调方案如何选择？
- 二、Full Fine Tuning vs Parameter-Efficient Fine-Tuning
- 三、Full Fine Tuning 篇
  - 3.1 介绍一下 Full Fine Tuning？
  - 3.2 介绍一下 Full Fine Tuning 优点？
  - 3.3 介绍一下 Full Fine Tuning 缺点？
- 四、Parameter-Efficient Fine-Tuning 篇
  - 4.1 介绍一下 Parameter-Efficient Fine-Tuning？
- 五、LoRA 篇
  - 5.1 介绍一下 LoRA？
  - 5.2 介绍一下 LoRA 流程？
  - 5.3 介绍一下 LoRA 优点？
  - 5.4 介绍一下 LoRA 缺点？
- 六、QLoRA 篇
  - 6.1 介绍一下 QLoRA？
  - 6.2 介绍一下 QLoRA 流程？
- ...

- [点击查看答案](https://articles.zsxq.com/id_e2piver2uzei.html)

## 七、大模型（LLMs）推理面 

### [大模型（LLMs）推理面](https://articles.zsxq.com/id_b9eecaoga75i.html)

- 1. 为什么大模型推理时显存涨的那么多还一直占着？
- 2. 大模型在gpu和cpu上推理速度如何？
- 3. 推理速度上，int8和fp16比起来怎么样？
- 4. 大模型有推理能力吗？
- ...

- [点击查看答案](https://articles.zsxq.com/id_b9eecaoga75i.html)

## 八、大模型（LLMs）增量预训练篇 

### [大模型（LLMs）增量预训练篇](https://articles.zsxq.com/id_jfq8la7g20ww.html)

- 1. 为什么要增量预训练？
- 2. 进行 增量预训练 需要做哪些准备工作？
- 3. 增量预训练 所用 训练框架？
- 4. 增量预训练 训练流程 是怎么样？
- ...

- [点击查看答案](https://articles.zsxq.com/id_jfq8la7g20ww.html)

### [增量预训练（Pretrain）样本拼接篇](https://articles.zsxq.com/id_8f35p8piwl4v.html)

- 一、 推理过程 分哪些阶段？
  - 1.1 Prefill（输入理解与初始化）阶段
  - 1.2 Decoding（递归推理与解码输出）阶段
- 二、推理性能的评价指标？
  - 2.1 Throughput（吞吐量）
  - 2.2 First Token Latency（首字延迟）
  - 2.3 Latency（延迟）
  - 2.4 QPS（每秒请求数）
- ...

- [点击查看答案](https://articles.zsxq.com/id_8f35p8piwl4v.html)

### [增量预训练（Pretrain）样本拼接篇](https://articles.zsxq.com/id_enteq22h1nhq.html)

- 一、Pretrain阶段，为什么需要拼接拼接？
- 二、有哪些 拼接方式？
  - 2.1 拼接方式一：Random Concatenate
  - 2.2 拼接方式二：Random Concatenate + NoiseMask
  - 2.3 拼接方式三：Random Concatenate + Cluster
  - 2.4 拼接方式四：IN-CONTEXT PRETRAINING

- [点击查看答案](https://articles.zsxq.com/id_enteq22h1nhq.html)

### [基于lora的llama2二次预训练](https://articles.zsxq.com/id_xo09u14omdjw.html)

- 一、为什么需要 对 llama2 做 基于lora的二次预训练?
- 二、基于lora的llama2二次预训练 的目标是什么？
- 三、基于lora的llama2二次预训练 的思想是什么？
- 四、基于lora的llama2二次预训练 语料构建思路？
- ...

- [点击查看答案](https://articles.zsxq.com/id_xo09u14omdjw.html)

## [九、大模型（LLMs）评测面](https://articles.zsxq.com/id_j9wcj62eovgc.html)

- 1 大模型怎么评测？
- 2 大模型的honest原则是如何实现的？模型如何判断回答的知识是训练过的已知的知识，怎么训练这种能力？
- 3 如何衡量大模型水平？
- 4 大模型评估方法 有哪些？
- ...

- [点击查看答案](https://articles.zsxq.com/id_j9wcj62eovgc.html)

## 十、大模型（LLMs）强化学习面 

### [大模型（LLMs）强化学习面](https://articles.zsxq.com/id_20xnfnoprj9s.html) 

- 1 简单介绍强化学习？
- 2 简单介绍一下 RLHF？
- 3 奖励模型需要和基础模型一致吗？
- 4 RLHF 在实践过程中存在哪些不足？
- 5 如何解决 人工产生的偏好数据集成本较高，很难量产问题？
- 6 如何解决三个阶段的训练（SFT-\>RM-\>PPO）过程较长，更新迭代较慢问题？
- 7 如何解决 PPO 的训练过程同时存在4个模型（2训练，2推理），对计算资源的要求较高 问题？
- 8 强化学习跟大语言模型的本质联系是什么？
- ...

- [点击查看答案](https://articles.zsxq.com/id_20xnfnoprj9s.html)

### [大模型（LLMs）强化学习——RLHF及其变种面](https://articles.zsxq.com/id_3ct6sw0wouna.html) 

- 一、介绍一下 LLM的经典预训练Pipeline？
- 二、预训练（Pre-training）篇
  - 2.1 具体介绍一下 预训练（Pre-training）？
- 三、有监督微调（Supervised Tinetuning）篇
  - 3.1 具体介绍一下 有监督微调（Supervised Tinetuning）？
  - 3.2 有监督微调（Supervised Tinetuning）的训练数据格式是什么样？
  - 3.3 预训练（Pre-training） vs 有监督微调（Supervised Tinetuning）区别？
- 四、对齐（Alignment）篇
  - 4.1 简单介绍一下 对齐（Alignment）？
- ...

- [点击查看答案](https://articles.zsxq.com/id_3ct6sw0wouna.html)

### [大模型（LLMs）强化学习—— PPO 面](https://articles.zsxq.com/id_s8kwqw1gowvh.html) 

  - 一、大语言模型RLHF中的PPO主要分哪些步骤？
  - 二、举例描述一下 大语言模型的RLHF？
  - 三、大语言模型RLHF 采样篇
    - 3.1 什么是 PPO 中 采样过程？
    - 3.2 介绍一下 PPO 中 采样策略？
    - 3.3 PPO 中 采样策略中，如何评估“收益”？
  - 四、在PPO过程中，reward model的效果上会有什么问题？
  - ...

- [点击查看答案](https://articles.zsxq.com/id_s8kwqw1gowvh.html)

### [RLHF平替算法DPO篇](https://articles.zsxq.com/id_mlq44r1p7nob.html) 

- RLHF平替算法DPO篇
  - 一、DPO vs RLHF？
  - 二、介绍一下 DPO的损失函数？
  - 三、DPO 微调流程 ?
  - 四、说一下 DPO 是如何简化 RLHF 的？
  - 五、DPO的第0步loss是固定的么？如果固定的话，值是多少？
  - 六、DPO是一个on-policy还是off-policy的算法，以及这样的算法有什么优劣？
  - 七、DPO公式是由PPO的objective公式推导过来的，为什么DPO是off-policy算法，而PPO是on-policy算法，到底哪一步推导出了问题？
  - ...

- [点击查看答案](https://articles.zsxq.com/id_mlq44r1p7nob.html)

### [reward 篇](https://articles.zsxq.com/id_vblb0j5qnaxg.html) 

  - 1 介绍一下 RM模型？
  - 2 为什么需要 RM模型？
  - 3 RM模型训练数据如何构建？
  - 4 reward 模型训练步骤中，为什么这一步骤在标注数据过程中不让人直接打分，而是去标排列序列呢?
  - 5 reward 模型的 loss 是怎么计算的?
  - ...

- [点击查看答案](https://articles.zsxq.com/id_vblb0j5qnaxg.html)

### [强化学习在自然语言处理下的应用篇](https://articles.zsxq.com/id_5tsn84l32eea.html) 

- 一、强化学习基础面
  - 1.1 介绍一下强化学习？
  - 1.2 介绍一下强化学习 的 状态（States） 和 观测（Observations）？
  - 1.3 强化学习 有哪些 动作空间（Action Spaces），他们之间的区别是什么？
  - ...

- [点击查看答案](https://articles.zsxq.com/id_5tsn84l32eea.html)

## 十一、大模型（LLMs）训练集面 

### [大模型（LLMs）训练集面](https://articles.zsxq.com/id_axtljtl0bsvw.html)

- SFT（有监督微调）的数据集格式？
- RM（奖励模型）的数据格式？
- PPO（强化学习）的数据格式？
- ...

- [点击查看答案](https://articles.zsxq.com/id_axtljtl0bsvw.html)

### [大模型（LLMs）LLM生成SFT数据方法面](https://articles.zsxq.com/id_1gzdghj84f9f.html)

- 四、大模型微调数据集格式篇
- 一、SFT数据集如何生成？
- 二、Self-Instruct 篇
  - ...

- [点击查看答案](https://articles.zsxq.com/id_1gzdghj84f9f.html)

## 十二、大模型（LLMs）显存问题面 

### [大模型（LLMs）显存问题面](https://articles.zsxq.com/id_jhiocx89p3su.html)

- 大模型大概有多大，模型文件有多大?
- 能否用4 * v100 32G训练vicuna 65b？
- 如果就是想要试试65b模型，但是显存不多怎么办？
- nB模型推理需要多少显存？
- ...

- [点击查看答案](https://articles.zsxq.com/id_jhiocx89p3su.html)

### [大模型（LLMs）显存优化策略篇](https://articles.zsxq.com/id_a1l60awgge6q.html)

- 一、介绍一下 gradient accumulation 显存优化方式？
- 二、介绍一下 gradient checkpointing 显存优化方式？

- [点击查看答案](https://articles.zsxq.com/id_a1l60awgge6q.html)

## 十三、大模型（LLMs）分布式训练面 

### [大模型（LLMs）分布式训练面](https://articles.zsxq.com/id_ah2ibj3z22c7.html)

- 1 理论篇
  - 1.1 训练 大语言模型 存在问题？
  - 1.2 什么是 点对点通信？
  - 1.3 什么是 集体通信？
  - 1.4 什么是 数据并行？
  - 1.5 数据并行 如何 提升效率？
  - 1.6 什么是 流水线并行？
  - 1.7 什么是 张量并行 (intra-layer)？
  - 1.8 数据并行 vs 张量并行 vs 流水线并行?
  - 1.9 什么是 3D并行？
  - 1.10 想要训练1个LLM，如果只想用1张显卡，那么对显卡的要求是什么？
  - 1.11 如果有N张显存足够大的显卡，怎么加速训练？
  - 1.12 如果显卡的显存不够装下一个完整的模型呢？
  - 1.13 PP推理时，是一个串行的过程，1个GPU计算，其他空闲，有没有其他方式？
  - 1.14 3种并行方式可以叠加吗？
  - 1.15 Colossal-AI 有1D/2D/2.5D/3D，是什么情况？
  - 1.16 除了3D并行有没有其他方式大规模训练？
  - 1.17 有了ZeRO系列，为什么还需要3D并行？
  - 1.18 平民适不适合玩3D并行？
  - 1.19 平民适不适合直接上多机多卡的ZeRO3（万兆网）？
  - 1.20 分布式并行及显存优化技术并行技术有哪一些，都有什么特点？
  - 1.21 显存优化技术有哪一些，都有什么特点？
  - 1.22 常见的分布式训练框架哪一些，都有什么特点？
- 2 实践篇
  - 2.1 假如有超多的8卡A100节点（DGX A100），如何应用3D并行策略？
  - 2.2 如果想构这样一个大规模并行训练系统，训练框架如何选？
  - 2.3 训练框架如何选？
- ...

- [点击查看答案](https://articles.zsxq.com/id_ah2ibj3z22c7.html)

### [图解分布式训练（一） —— 流水线并行（Pipeline Parallelism）面](https://articles.zsxq.com/id_wre1eni0oq7d.html)

- 为什么需要流水线并行（Pipeline Parallelism）？
- 一、流水线并行（Pipeline Parallelism） 优化目标是什么？
- ...

- [点击查看答案](https://articles.zsxq.com/id_wre1eni0oq7d.html)

### [图解分布式训练（二） —— nn.DataParallel面](https://articles.zsxq.com/id_9dfwi0ooio2z.html)

- 为什么需要nn.DataParallel？
- 一、pytorch中的GPU操作默认是什么样？
- 二、介绍一下 nn.DataParallel 函数？
- 三、nn.DataParallel 函数 处理逻辑 介绍一下？
- ...

- [点击查看答案](https://articles.zsxq.com/id_9dfwi0ooio2z.html)

### [图解分布式训练（三） ——  nn.parallel.DistributedDataParallel](https://articles.zsxq.com/id_i4s3ia057rmh.html)

- 为什么需要 nn.parallel.DistributedDataParallel ？
- 一、什么是 DistributedDataParallel 核心 —— Ring-AllReduce？
- 二、nn.parallel.DistributedDataParallel 函数 介绍一下？
- 三、nn.parallel.DistributedDataParallel 函数 如何多卡加速训练？
- ...

- [点击查看答案](https://articles.zsxq.com/id_i4s3ia057rmh.html)

### [图解分布式训练（四） ——  torch.multiprocessing 详细解析](https://articles.zsxq.com/id_gu9smpbn510e.html)

- 一、torch.multiprocessing 函数介绍一下？
- 二、torch.multiprocessing 函数如何使用？
- ...

- [点击查看答案](https://articles.zsxq.com/id_gu9smpbn510e.html)

### [图解分布式训练（五） ——  AMP混合精度训练 详细解析](https://articles.zsxq.com/id_0slrgoti6gvb.html)

- 为什么需要 AMP混合精度训练？
- 一、什么是自动混合精度训练(AMP)
- 二、为什么需要自动混合精度？
- 三、混合精度训练的优点是什么？
- ...

- [点击查看答案](https://articles.zsxq.com/id_0slrgoti6gvb.html)

### [图解分布式训练（六） —— Pytorch的 DeepSpeed 详细解析](https://articles.zsxq.com/id_kmq9rn2vo4kz.html)

- 一、为什么需要 Deepspeed？
- 二、DeepSpeed 基本概念 介绍一下？
  - 2.1 DeepSpeed 介绍
  - 2.2 DeepSpeed 基础的概念
  - 2.3 DeepSpeed 支持的功能
- 三、DeepSpeed 通信策略 介绍一下？
- 四、DeepSpeed 如何使用？
  - 4.1 DeepSpeed 安装
  - 4.2 DeepSpeed 使用
- ...

- [点击查看答案](https://articles.zsxq.com/id_kmq9rn2vo4kz.html)

### [图解分布式训练（七）—— accelerate 分布式训练 详细解析](https://articles.zsxq.com/id_o5wkeionnqr7.html)

- 一、为什么需要 accelerate 分布式训练？
- 二、什么是 accelerate 分布式训练?
- ...

- [点击查看答案](https://articles.zsxq.com/id_o5wkeionnqr7.html)

### [图解分布式训练（八）—— ZeRO 学习](https://articles.zsxq.com/id_grv7uddls2g1.html)

- 一、什么是 3D 并行？
- 二、3D 并行 策略有哪些？
- 三、为什么需要 ZeRO？
- ...

- [点击查看答案](https://articles.zsxq.com/id_grv7uddls2g1.html)

### [大模型分布式训练故障恢复篇](https://articles.zsxq.com/id_zspm2q33tckx.html)

- 一、为什么 大模型分布式训练 需要 故障恢复？
- 二、如何获取最优的ckpt存储间隔？
- 三、ckpt存储能否实现异步或者部分掩盖？
- ...

- [点击查看答案](https://articles.zsxq.com/id_zspm2q33tckx.html)

### [图解分布式训练（九）—— Megatron-LM 篇](https://articles.zsxq.com/id_o4qtcspmuwqv.html)

- 1、Activation Recomputation是怎么实现的?
- 2、Megatron中的OverlappedDistributed Optimizer 是如何实现的?
- 3、Megatron-LM 中 Context Parallel 篇
  - 3.1 介绍一下 Megatron-LM 中 Context Parallel 实现原理？
  - ...

- [点击查看答案](https://articles.zsxq.com/id_o4qtcspmuwqv.html)

### [分布式训练 Trick 汇总篇](https://articles.zsxq.com/id_fu9065izm2m4.html)

- 一、数据并行 Trick 篇
  - 1.1 数据并行 FSDP
  - 1.2 数据并行 DDP
  - 1.3 数据并行 ZeRO
    - 1.3.1 Model state
    - 1.3.2 Residual state
    - 1.3.3 offload
- ...

- [点击查看答案](https://articles.zsxq.com/id_fu9065izm2m4.html)

### [pytorch 分布式计算 坑/bug 梳理篇](https://articles.zsxq.com/id_onztfzwdckom.html)

- 一、使用 DistributedDataParallel（分布式并行）时，显存分布不均衡问题
- 二、如果是用pytorch实现同步梯度更新，自研 数据接口，出现 第一个epoch结尾处程序卡死问题
- ...

- [点击查看答案](https://articles.zsxq.com/id_onztfzwdckom.html)

## 十四、大模型（LLMs）agent 面

### [大模型（LLMs）agent 面](https://articles.zsxq.com/id_le02luntesap.html) 

- 一、什么是 大模型（LLMs）agent？
- 二、大模型（LLMs）agent 有哪些部分组成？
  - 2.1 介绍一下 规划（planning）？
    - 2.1.1 拆解子目标和任务分解
      - 2.1.1.1 如何进行 拆解子目标和任务分解？
      - 2.1.1.2 拆解子目标和任务分解 有哪些方法？
    - 2.1.2 模型自我反省
      - 2.1.2.1 如何进行 模型自我反省？
      - 2.1.2.2 模型自我反省 有哪些方法？
  - 2.2 介绍一下 记忆（Memory）？
  - 2.3 介绍一下 工具使用（tool use）？
- 三、大模型（LLMs）agent 主要 利用了 大模型 哪些能力？
- ...

- [点击查看答案](https://articles.zsxq.com/id_le02luntesap.html)

### 函数调用 Function Call 篇

- [函数调用 Function Call 篇](https://articles.zsxq.com/id_asxg09gtrx89.html)
  - 一、为什么需要 函数调用(function call)？
  - 二、什么是 函数调用(function call)？
  - ...

- [点击查看答案](https://articles.zsxq.com/id_asxg09gtrx89.html)

- [开源模型 Function Call 篇](https://articles.zsxq.com/id_s2ojkzdw83gb.html)
  - 开源模型 Function Call 方案有哪些？
    - Llama 3.1
      - 对话协议（Chat Protocal）
      - Tool Call Template 样式
      - ...

- [点击查看答案](https://articles.zsxq.com/id_s2ojkzdw83gb.html)

## [十五、LLMs 位置编码篇](https://articles.zsxq.com/id_amt4qkusdcir.html) 

- 一、什么是位置编码？
- 二、为什么需要位置编码？
- 三、什么是绝对位置编码？
  - 3.1 训练式位置编码篇
    - ...
- 四、什么是相对位置编码？
- 五、旋转位置编码 RoPE篇
  - 5.1 旋转位置编码 RoPE 思路是什么？
  - ...
- 六、长度外推问题篇
  - 6.1 什么是 长度外推问题？
  - 6.2 长度外推问题 的 解决方法 有哪些？
- 七、 ALiBi (Attention with Linear Biases)篇
  - 7.1 ALiBi (Attention with Linear Biases) 思路是什么？
  - ...

- [点击查看答案](https://articles.zsxq.com/id_amt4qkusdcir.html)

## 十六、LLMs Tokenizer 篇

### [LLMs Tokenizer 篇](https://articles.zsxq.com/id_6z8ptdwqeid7.html)

- LLMs Tokenizer 篇
  - Byte-Pair Encoding(BPE)篇
    - 1 介绍一下 Byte-Pair Encoding(BPE) ？
    - 2 Byte-Pair Encoding(BPE) 如何构建词典？
    - 3 Byte-Pair Encoding(BPE) 具有什么优点？
    - 4 Byte-Pair Encoding(BPE) 具有什么缺点？
    - 5 手撕 Byte-Pair Encoding(BPE) ？
  - Byte-level BPE 篇
    - 1 介绍一下 Byte-level BPE ？
    - 2 Byte-level BPE 如何构建词典？
    - 3 Byte-level BPE 具有什么优点？
    - 4 Byte-level BPE 具有什么缺点？
  - WordPiece 篇
    - ...

- [点击查看答案](https://articles.zsxq.com/id_6z8ptdwqeid7.html)

### [怎么让英文大语言模型支持中文？（一） —— 构建中文tokenization](https://articles.zsxq.com/id_w0d2q29sueq7.html)

- 一、为什么需要 构建中文tokenization？
- 二、如何对 原始数据预处理？
- 三、如何构建中文的词库？
- ...

- [点击查看答案](https://articles.zsxq.com/id_w0d2q29sueq7.html)

### [怎么让英文大语言模型支持中文？（二） —— 继续预训练篇](https://articles.zsxq.com/id_jprkwhrvf3tw.html)

- 一、为什么需要进行继续预训练？
- 二、如何对 继续预训练 数据预处理？
- 三、如何 构建模型？
- 四、如何 使用模型？

- [点击查看答案](https://articles.zsxq.com/id_jprkwhrvf3tw.html)

### [怎么让英文大语言模型支持中文？（三） —— 对预训练模型进行指令微调](https://articles.zsxq.com/id_p2wj7zadwxwb.html)

- 一、为什么需要对预训练模型进行指令微调？
- 二、对预训练模型进行指令微调 数据 如何处理？
- 三、对预训练模型进行指令微调 tokenization 如何构建？
- 四、对预训练模型进行指令微调 模型 如何构建？
- 五、是否可以结合 其他库 使用？

- [点击查看答案](https://articles.zsxq.com/id_p2wj7zadwxwb.html)

## 十七、大模型（LLMs）加速篇 

### [大模型(LLM)部署框架对比篇](https://articles.zsxq.com/id_7d31dgh26fcp.html)

- 大模型(LLM)部署框架对比篇
- 一、为什么需要对大模型推理加速？
- 二、大模型(LLM)部署框架对比总览
- 三、大模型(LLM)部署优化策略
  - ...

- [点击查看答案](https://articles.zsxq.com/id_7d31dgh26fcp.html)

### [大模型（LLMs）推理加速篇](https://articles.zsxq.com/id_kgzsxgro8cee.html)

- 一、 推理过程 分哪些阶段？
    - 1.1 Prefill（输入理解与初始化）阶段
    - 1.2 Decoding（递归推理与解码输出）阶段
- 二、 推理性能的评价指标？
    - 2.1 Throughput（吞吐量）
    - 2.2 First Token Latency（首字延迟）
    - 2.3 Latency（延迟）
    - 2.4 QPS（每秒请求数）
- 三、 当前优化模型最主要技术手段有哪些？
    - ...

- [点击查看答案](https://articles.zsxq.com/id_kgzsxgro8cee.html)


### [大模型（LLMs）加速篇](https://articles.zsxq.com/id_w9wewc152eux.html)

- 1 当前优化模型最主要技术手段有哪些？
- 2 推理加速框架有哪一些？都有什么特点？
- 3 vLLM 篇
  - 3.1 vLLM 的 功能有哪些？
  - ...

- [点击查看答案](https://articles.zsxq.com/id_w9wewc152eux.html)

### [LLMs 推理性能面](https://articles.zsxq.com/id_jwd03u0l7feo.html) 

- 一、介绍一下 LLMs 的文本生成过程？
- 二、如何准确衡量模型的推理速度呢？
- 三、如果对整体推理时延有具体目标，有哪些有效的启发式方法来评估模型？
- ...

- [点击查看答案](https://articles.zsxq.com/id_jwd03u0l7feo.html)

### [LLM（大语言模型）部署加速方法——PagedAttention篇](https://articles.zsxq.com/id_p22mjq881n3n.html)

- 一、vLLM 用于大模型并行推理加速 存在什么问题？
- 二、vLLM 如何 优化 大模型并行推理加速？
- 三、什么是 PagedAttention？
- ...

- [点击查看答案](https://articles.zsxq.com/id_p22mjq881n3n.html)

### [大模型推理加速工具 —— vLLM](https://articles.zsxq.com/id_zw5h9ogvac2w.html)

- 一、引言
  - 1.1 前言
  - 1.2 为什么 需要 vLLM ?
  - 1.3 vLLM 具有哪些特点 ?
  - 1.4 vLLM 支持哪些 Huggingface 模型 ?
- 二、vLLM 性能如何？
- ...

- [点击查看答案](https://articles.zsxq.com/id_zw5h9ogvac2w.html)

### [LLM（大语言模型）部署加速方法——Faster Transformer篇](https://articles.zsxq.com/id_dd2gowztxtfg.html)

- 一、为什么需要 FasterTransformer？
- 二、FasterTransformer 介绍一下？
- 三、FasterTransformer 核心是什么？
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
- ...

- [点击查看答案](https://articles.zsxq.com/id_9a643feq2b0b.html)

### [LLM推理技术之StreamingLLM：如何拥有无限长生成能力](https://articles.zsxq.com/id_w1gwi9z7qm5s.html)

- 一、前言
  - 1.1 大型语言模型（LLM）存在什么问题？
  - 1.2 StreamingLLM 背景介绍
  - 1.3 StreamingLLM 核心问题？
  - ...
- 二、StreamingLLM 的思路是什么？
- ...

- [点击查看答案](https://articles.zsxq.com/id_w1gwi9z7qm5s.html)

### [SwiftInfer —— 大模型无限流式输入推理飙升46%，打破多轮对话长度限制](https://articles.zsxq.com/id_0rpua5fejfwc.html) 

- StreamingLLM 篇
  - 一、为什么需要 StreamingLLM？
  - 二、StreamingLLM 思路是什么？
  - 三、StreamingLLM 优点是什么？
- SwiftInfer 篇：基于TensorRT的StreamingLLM实现
  - ...

- [点击查看答案](https://articles.zsxq.com/id_0rpua5fejfwc.html)

## 十八、大模型幻觉（LLM Hallucination）面 

### [大模型幻觉（LLM Hallucination）面](https://articles.zsxq.com/id_schwrdmvmhr7.html)

- 一、什么是大模型幻觉？
- 二、为什么LLM会产生幻觉？
- 三、为什么需要解决LLM的幻觉问题？
- 四、幻觉一定是有害的吗？
- ...

- [点击查看答案](https://articles.zsxq.com/id_schwrdmvmhr7.html)

### [大模型的幻觉问题篇](https://articles.zsxq.com/id_8mr4mlhe5q1x.html)

- 一、什么是 大模型幻觉问题？
- 二、为什么 会 出现 大模型幻觉问题？
- ...

- [点击查看答案](https://articles.zsxq.com/id_8mr4mlhe5q1x.html)

### [如何缓解大模型幻觉？](https://articles.zsxq.com/id_tbezgzifowzp.html)

- 一、为什么 会 出现 大模型幻觉？
- 二、如何 缓解 大模型幻觉？

- [点击查看答案](https://articles.zsxq.com/id_tbezgzifowzp.html)

## 十九、LLMs 对比篇 

### [LLMs 对比篇](https://articles.zsxq.com/id_fsq8czgwjxse.html)

- LLMs 对比篇
  - 一、谈谈你对当前出现的各种大模型的见解？
  - 二、目前大模型常见的 base 模型训练和 chat 模型训练 方式 的区别么？
  - 三、llama、baichuan、ChatGLM、Bloom 和 qwen 等开源大模型技术对比篇
    - 3.1 llama 系列篇
      - 3.1.1 llama 篇
        - 3.1.1.1 llama 训练数据 介绍
        - 3.1.1.2 llama 模型参数量 介绍
        - 3.1.1.3 llama 模型结构 介绍
        - 3.1.1.4 llama 训练目标 介绍
        - 3.1.1.5 llama tokenizer 介绍
        - 3.1.1.6 llama 衍生模型 介绍
        - 3.1.1.7 llama 词表扩展: Chinese LLaMA
      - 3.2.1 llama2 篇
        - 3.2.1 llama2 系列 数据预处理方式？
        - 3.2.2 llama2 系列 Tokenizer 处理方式？
        - 3.2.3 llama2 系列 Architectural？
        - 3.2.4 llama2 系列 content长度？
    - 3.2 Mistral 7B 系列篇
      - 3.2.1  Mistral 7B Architectural？
    - 3.3 Qwen 系列篇
      - 3.3.1 Qwen 系列 数据预处理方式？
      - 3.3.2 Qwen 系列 Tokenizer 处理方式？
      - 3.3.3 Qwen 系列 ARCHITECTURE？
    - 3.4 Baichuan 系列篇
      - 3.4.1 Baichuan2 篇
        - 3.4.1.1 Baichuan2 系列 数据预处理方式？
        - 3.4.1.2 Baichuan2 系列 Tokenizer 处理方式？
        - 3.4.1.2 Baichuan2 系列 Architecture ？
    - 3.5 GLM 系列篇
      - 3.5.1 ChatGLM-6B 篇
        - 3.5.1.1 ChatGLM-6B 结构特点？
        - 3.5.1.2 ChatGLM-6B 训练目标？
        - 3.5.1.3 ChatGLM-6B  tokenizer？
    - 3.6 BLOOM 系列篇
      - 3.6.1 BLOOM 篇
        - 3.6.1.1 BLOOM 训练数据构建？
        - 3.6.1.2 BLOOM 模型参数量？
        - 3.6.1.3 BLOOM 模型结构？
        - 3.6.1.4 BLOOM 训练目标？
        - 3.6.1.5 BLOOM tokenizer?
  - 四、分析与总结？
    - 4.1 大模型训练共同点？
    - 4.2 大模型训练不同点？
  - 五、对比
    - 5.1 LLaMA、ChatGLM 和 BLOOM 对比
    - 5.2 LLaMA、ChatGLM 和 BLOOM 的 tokenizer 比较
    - 5.3LLaMA、ChatGLM 和 BLOOM 的 结果 比较

- [点击查看答案](https://articles.zsxq.com/id_fsq8czgwjxse.html)

### [LLMs 对比篇](https://articles.zsxq.com/id_0j7k3gxa5hpm.html)

- 大模型-attention mask 篇
  - 1、prefix-tuning的prefix tokens是双向注意力吗？
  - 2、chatglm1和chatglm2的attention mask是怎么样的？
  - 3、llama的attention mask是怎么样的？

- [点击查看答案](https://articles.zsxq.com/id_0j7k3gxa5hpm.html)

### [百川智能baichuan7B、13B、53B、baichuan2 总结篇](https://articles.zsxq.com/id_ma6pw7v2g9pi.html)

- 一、baichuan-7B篇
  - 1. 你了解baichuan-7B解构么？介绍一下？
  - 2. baichuan-7B 如何 收集原始数据并 构建 训练数据？
  - 3. baichuan-7B 如何 提高 训练稳定性和吞吐？
- 二、baichuan-13B篇
  - ...

- [点击查看答案](https://articles.zsxq.com/id_ma6pw7v2g9pi.html)

### [LLaMa 篇](https://articles.zsxq.com/id_9ba6a72wan2w.html) 

- 一、相比较于llama而言，llama2有哪些改进，对于llama2是应该如何finetune？

- [点击查看答案](https://articles.zsxq.com/id_9ba6a72wan2w.html)

### [GPT 经验篇](https://articles.zsxq.com/id_r46k6bqu34xh.html) 

- 一、gpt源码past\_key\_value是干啥的？
- 二、gpt onebyone 每一层怎么输入输出？
- 三、bert和gpt有什么区别
- 四、文本生成的几大预训练任务？
- 五、讲讲T5和Bart的区别，讲讲bart的DAE任务？
- 六、讲讲Bart和Bert的区别？
- 七、gpt3和gpt2的区别？

- [点击查看答案](https://articles.zsxq.com/id_r46k6bqu34xh.html)

## 二十、思维链 Chain-of-Thought（COT）篇 

### [思维链 Chain-of-Thought（COT）篇](https://articles.zsxq.com/id_c0jpjo7q95wg.html)

- 一、什么是思维链提示？
- 二、思维链提示本质是什么？
- 三、思维链提示 与 标准的提示学习方法有什么不同?
- 四、思维链提示 为什么可以提高语言模型的复杂推理能力?它的优势在哪里?
- ...

- [点击查看答案](https://articles.zsxq.com/id_c0jpjo7q95wg.html)

### [思维链 Chain-of-Thought（COT）变体篇](https://articles.zsxq.com/id_thdljw9vgxt1.html)

- 思维链 Chain-of-Thought（COT）：思维链的启蒙
  - 1. 什么是 思维链 Chain-of-Thought（COT）？
  - 2. 思维链 Chain-of-Thought（COT）是思路是什么？
  - 3. 思维链 Chain-of-Thought（COT）存在问题？
- 思维树 Tree of Thoughts（TOT）：一种用树结构解决复杂问题的方法
  - 1. 为什么需要 思维树 Tree of Thoughts（TOT）？
  - 2. 什么是 思维树 Tree of Thoughts（TOT）？
  - 3. 思维树 Tree of Thoughts（TOT）涉及问题有哪些？
- ...

- [点击查看答案](https://articles.zsxq.com/id_thdljw9vgxt1.html)

### [小样本提示学习篇](https://articles.zsxq.com/id_re6ap2lq88gw.html) 

- 一、什么是Zero-shot提示方法？
- 二、什么是Few-shot提示方法？
- 三、阐述One-shot和Few-shot提示策略及其应用场景？
- 四、什么是逐步Zero-shot
- 五、定义Zero-shot-CoT提示策略并描述其应用方法？
- 六、解释Few-shot-CoT提示策略及其实际使用方式？
- 七、Few-shot-LtM策略包含哪些主要阶段及其职责？

- [点击查看答案](https://articles.zsxq.com/id_re6ap2lq88gw.html)

## [二十一、LLMs 测试集 中 数据泄露 问题篇](https://articles.zsxq.com/id_6e3k0i8x5ggm.html)

- 一、什么是 LLMs 测试集数据泄露 问题？
- 二、如何解决 LLMs 测试集数据泄露 问题？
- 三、是否可以 避开训练集来处理 LLMs 测试集数据泄露 问题？
  - ...

- [点击查看答案](https://articles.zsxq.com/id_6e3k0i8x5ggm.html)

## [二十二、MOE（Mixture-of-Experts）篇](https://articles.zsxq.com/id_w6ebwrhprpj9.html)

### 22.1 [MOE（Mixture-of-Experts）篇](https://articles.zsxq.com/id_5anfhj9qoh2v.html)

- 一、为什么需要 MOE（Mixture-of-Experts）？
- 二、MOE（Mixture-of-Experts）的思路是什么样的？
- 三、介绍一下 MOE（Mixture-of-Experts）分布式并行策略？
  - 3.1 MOE + 数据并行?
  - 3.2 MOE + 模型并行?
- 四、MoE大模型具备哪些优势？
- 五、MoE大模型具备哪些缺点？
- ...

- [点击查看答案](https://articles.zsxq.com/id_5anfhj9qoh2v.html)

### 22.2 [MOE大模型对比篇](https://articles.zsxq.com/id_j51bnu3xfgm9.html)

- DeepSpeed-MoE
- PAI-Megatron-Patch MoE
  
- [点击查看答案](https://articles.zsxq.com/id_j51bnu3xfgm9.html)

## 二十三、大模型蒸馏篇

### [大模型蒸馏篇](https://articles.zsxq.com/id_jkiw9vhzopgv.html)

- 一、知识蒸馏和无监督样本训练？
- 二、对知识蒸馏知道多少，有哪些改进用到了？
- 三、谈一下对模型量化的了解？
- ...

- [点击查看答案](https://articles.zsxq.com/id_jkiw9vhzopgv.html)

### [LLMs 浮点数篇](https://articles.zsxq.com/id_vu744g6jklli.html) 

- 一、fp32和fp16的区别，混合精度的原理
- 二、半精度是什么？
- 三、半精度的理论原理是什么？
- ...

- [点击查看答案](https://articles.zsxq.com/id_vu744g6jklli.html)

### [自定义 CUDA 函数的轻量级包装器 —— bitsandbytes篇](https://articles.zsxq.com/id_2nwi4napgvlh.html) 

- 一、什么是 bitsandbytes?
- 二、如何才能使用 bitsandbytes？
- 三、如何使用 bitsandbytes？
- ...

- [点击查看答案](https://articles.zsxq.com/id_2nwi4napgvlh.html)

## [二十四、大模型（LLMs）软硬件配置面](https://articles.zsxq.com/id_m5q8zk3wo84k.html)

- 建议的软件环境是什么？
- ...

- [点击查看答案](https://articles.zsxq.com/id_m5q8zk3wo84k.html)

## [二十五、Token及模型参数准备篇](https://articles.zsxq.com/id_9oplu4014qx5.html)

- 预训练数据 Token 重复 是否影响 模型性能？
- SFT需要训练Token数？

- [点击查看答案](https://articles.zsxq.com/id_9oplu4014qx5.html)

## 二十六、多模态常见面试篇

### [多模态常见面试篇](https://articles.zsxq.com/id_hmoqafrxjumk.html)

- 一、最近关注的论文，多模态视觉大模型(CLIP,DALLE)？
- 二、blip2的架构，优势和之前多模态模型的区别？
- ...

- [点击查看答案](https://articles.zsxq.com/id_hmoqafrxjumk.html)

## 二十七、NLP常见面试篇

### [NLP Trick 篇](https://articles.zsxq.com/id_bnzc5w57w7ox.html) 

- 一、怎么处理类别不平衡？
- 二、有了解其他模型去尝试解决长度限制的方案吗？
- ...

- [点击查看答案](https://articles.zsxq.com/id_bnzc5w57w7ox.html)

### [文本分类常见面试篇](https://articles.zsxq.com/id_fku4xbzkano0.html) 

- 一、文本分类任务有哪些应用场景？
- 二、文本分类的具体流程？
- 三、fastText的分类过程？fastText的优点？
- ...

- [点击查看答案](https://articles.zsxq.com/id_fku4xbzkano0.html)

### [文本摘要常见面试篇](https://articles.zsxq.com/id_gw097zgji66q.html) 

- 一、抽取式摘要和生成式摘要存在哪些问题？
- 二、Pointer-generator network解决了什么问题？
- 三、文本摘要有哪些应用场景？
- ...

- [点击查看答案](https://articles.zsxq.com/id_gw097zgji66q.html)

### [命名实体识别常见面试篇](https://articles.zsxq.com/id_2nueuvwwm7v0.html) 

- 一、CRF 常见面试题
  - 1.1 什么是CRF？CRF的主要思想是什么？
  - 1.2 CRF的三个基本问题是什么？
  - 1.3 线性链条件随机场的参数化形式？
  - 1.4 CRF的优缺点是什么？
  - 1.5 HMM与CRF的区别？
  - 1.6 生成模型与判别模型的区别？
- 二、HMM 常见面试题
  - ...

- [点击查看答案](https://articles.zsxq.com/id_2nueuvwwm7v0.html)

### [向量检索常见面试篇](https://articles.zsxq.com/id_dnq0o4aicjso.html) 

- 一、向量检索库总结
  - 1.1 Annoy
    - 1.1.1 Annoy 介绍
    - 1.1.2 Annoy 使用
  - 1.2 Faiss
    -...

- [点击查看答案](https://articles.zsxq.com/id_dnq0o4aicjso.html)

## 二十八、其他常见面试篇

### [LLMs 其他 Trick](https://articles.zsxq.com/id_958pher9zdxp.html)

1. huggingface 下载不了模型问题？
2. ...

- [点击查看答案](https://articles.zsxq.com/id_958pher9zdxp.html)

## 二十九、大模型推理加速——KV Cache篇

### [大模型推理加速——KV Cache篇](https://articles.zsxq.com/id_swmfcls3sp1j.html)

- 大模型推理加速——KV Cache篇
  - 一、介绍一下 KV Cache是啥？
  - 二、为什么要进行 KV Cache？
    - 2.1 不使用 KV Cache 场景
    - 2.2 使用 KV Cache 场景
  - 三、说一下 KV Cache 在 大模型中的应用？
    - ...

- [点击查看答案](https://articles.zsxq.com/id_swmfcls3sp1j.html)

## 三十、大模型——角色扮演大模型篇

### [大模型——角色扮演大模型篇](https://articles.zsxq.com/id_16kl2onmsf8t.html)

- 大模型——角色扮演大模型篇
  - 一、什么是角色扮演大模型？
  - 二、为什么需要角色扮演大模型？
  - 三、角色扮演大模型 相比于 通用大模型 具有哪些区别？
  - 四、能否通俗易懂的介绍 【角色扮演大模型】？
  - ...

- [点击查看答案](https://articles.zsxq.com/id_16kl2onmsf8t.html)

## 三十一、大模型——Chat o1 篇

### [千面郎君 篇（三十一章）—— OpenAI o1 篇](https://articles.zsxq.com/id_71rmw7acx3cd.html)

- 千面郎君 篇（三十一章）—— OpenAI o1 篇
  - 一、Shortcut learning (捷径学习) vs Journey learning (旅程学习)
    - 1.1 Shortcut learning (捷径学习)
      - 1.1.1 什么是 Shortcut learning (捷径学习)？
      - 1.1.2 Shortcut learning (捷径学习) 包含哪些关键特征？
      - 1.1.3 Shortcut learning (捷径学习) 优点是什么？
      - 1.1.4 Shortcut learning (捷径学习) 缺点是什么？
    - 1.2 Journey learning (旅程学习)
      - 1.2.1 什么是 Journey learning (旅程学习)？
      - 1.2.2 Journey learning (旅程学习) 包含哪些关键特征？
      - 1.2.3 Journey learning (旅程学习) 优点是什么？
    - 1.3 Shortcut learning (捷径学习) vs Journey learning (旅程学习)
  - 二、o1 的长思维链篇
    - 2.1 o1 的长思维链是什么样子？
    - 2.2 长思维 (Long thought) 是如何工作的？
    - 2.3 如何构建长思维？
  - ...

- [点击查看答案](https://articles.zsxq.com/id_71rmw7acx3cd.html)

### [OpenAI o1 面试篇](https://articles.zsxq.com/id_032nwgcgwhc6.html)

- OpenAI o1 面试篇
  - Q: o1 的训练方法与之前的模型有何主要区别？
  - Q: o1 的"思考"过程与简单的提示有何不同？
  - Q: 为什么 o1 在推理任务上比之前的模型更强大？
  - Q: o1 如何处理安全性问题？
  - ...

- [点击查看答案](https://articles.zsxq.com/id_032nwgcgwhc6.html)

### [Scaling LLM Test-Time：谁说类o1推理一定要用RL?](https://articles.zsxq.com/id_71l9woqohebk.html)

- Scaling LLM Test-Time：谁说类o1推理一定要用RL?
  - 一、Scaling LLM Test-Time 介绍篇
    - 1.1 为什么需要 Scaling LLM Test-Time？
    - 1.2 三种 Scaling LLM Test-Time 类型定义？
    - 1.3 有哪些 Scaling Test-Time的方法？
    - 问题引申
  - 二、方法一：纯 Inference Scaling 篇
    - 2.1 Inferece Test-Time的统一视角：Proposer \& Verifier
    - 2.2 Proposer \& Verifier 实例：Best-of-N
    - ...

- [点击查看答案](https://articles.zsxq.com/id_71l9woqohebk.html)

