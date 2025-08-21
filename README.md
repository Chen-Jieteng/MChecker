# MChecker
## 项目描述
开源机审模型全流程项目，模拟针对通用的APP短视频、评论、广告等内容的机审模型架构的实践及优化。（这里写一句话）以提供现有审核平台一个新路径及思路。
此项目是全网第一个开源审核平台工作流自动化项目，同时包含自动审核内容、自动数据标注、自动拟人填写审批表单、自动生成各类专业文档等。
自动化审批功能有两种模式：监督模式、无监督模式

简单来说，就是将通用的传统审核页面、内容数据标注页面、模型使用情况管理系统、人工审批表单填写模块投射在同一个屏幕上。
一个机审运营人员对智能体的运维即可同时完成内容审核员、数据标注员、模型性能监控员、文档撰写专员的完整工作流程。


- 被测试APP为开源项目douyin（抖音）：https://github.com/zyronon/douyin
- 本项目对douyin的页面进行了增删查改。
- 其余部分（审核员控制面板、报告生成系统、后端服务器代码、MChecker智能体）为原创。


项目可能会有业务及专业性上的偏差，作者属技术背景，业务能力仍然无限的上升潜力，若有错误请通过Issues指正，谢谢大家。

## 项目架构
![进度](https://github.com/Chen-Jieteng/MChecker/blob/main/readme_images/%E4%B8%8B%E8%BD%BD.png)

## 审核流程demo：
（视频）
解释



## 审核平台页面
- 页面全貌（分为四个部分，左上为APP显示部分，右上为机审内容部分，左下为数据标注部分，右下为人工审批部分）
<img src="https://github.com/Chen-Jieteng/MChecker/blob/main/readme_images/%E9%A6%96%E9%A1%B5.png" width="60%">

- APP显示部分：以短视频为例，左边（短视频），右边（评论区）
<img src="https://github.com/Chen-Jieteng/MChecker/blob/main/readme_images/APP%E9%83%A8%E5%88%86.png" width="40%">

模型监控：
- 机审内容部分：实时状态、三种模型的版本管理面板（可以切换实验模型），实时报警模块、三个线上部署模型的数据面板（版本号，部署时间，模型类型，状态，平均准确率,延迟率，mAP，处理个数，GPU占用比例）
- 目标检测准确率，推理延迟，检测对象数，违规片段
- 严重程度占比（低，中，高，封禁）
- 置信度分布图
- 近7天的模型评估（准确率，精确度，F1值，召回率，AUC，微观AUC，宏观AUC，MAP@0.5, MAP@0.75, MAP@0.5:0.95, AVG-IOU，IOU-Threshold，FPS，时延，吞吐量，风险检测值，FPR，FNR，宏观F1，加权F1，Top1准确率，Top3准确率，Top5准确率，AVG-CONF，CONF-Threashold，平衡，Cohen-K值，Matthews-P值）
<img src="https://github.com/Chen-Jieteng/MChecker/blob/main/readme_images/%E6%9C%BA%E5%AE%A1%E7%9B%91%E6%8E%A7.png" width="100%">
- 模型触发规则，包括视觉规则，语音规则，文本规则，编号，描述，权重值
- 提示词配置：
  -- 视觉模型提示词：[提示词1](https://github.com/Chen-Jieteng/MChecker/blob/main/prompt/CV_prompt.md)
  -- 文本模型提示词：[提示词2](https://github.com/Chen-Jieteng/MChecker/blob/main/prompt/NLP_prompt.md)
  -- 语音模型提示词：[提示词3](https://github.com/Chen-Jieteng/MChecker/blob/main/prompt/speech_prompt.md)
- 提示此版本管理和性能分析

推理过程日志：
- 推理过程
- 推理设置：执行步数上线（默认15），频率默认2秒，是否启动ASR，采样温度（默认0.3），top_p值默认0.9
<img src="https://github.com/Chen-Jieteng/MChecker/blob/main/readme_images/%E6%8E%A8%E7%90%86%E8%BF%87%E7%A8%8B%E6%97%A5%E5%BF%97.png" width="60%">


数据标注部分：
- 数据标注 
<img src="https://github.com/Chen-Jieteng/MChecker/blob/main/readme_images/%E6%95%B0%E6%8D%AE%E6%A0%87%E6%B3%A8.png" width="100%">


人工审批部分
- 人工审核部分
<img src="https://github.com/Chen-Jieteng/MChecker/blob/main/readme_images/%E4%BA%BA%E5%B7%A5%E5%AE%A1%E6%A0%B8.png" width="100%">
 
- 审核结果窗口
  

- 文档生成
<img src="https://github.com/Chen-Jieteng/MChecker/blob/main/readme_images/%E6%99%BA%E8%83%BD%E6%96%87%E6%A1%A3%E7%94%9F%E6%88%90.png" width="60%">
生成的文档示例：产品经理PDE文件



## 智能体模式
- 单次测试：
- 监督处理：
- 无监督处理：


## 技术选型
模型层：
- 视觉模型（A/B测试组）：Qwen-VL-Plus（A组）, QVQ-Plus（B组）
- 文字推理模型（A/B测试组）：Qwen-Flash（A组），QWQ-Plus-Latest（B组）
- 语音识别模型（A/B测试组）：Paraformer-Realtime-8k（A组），Qwen-Audio-ASR（B组）
- 多模态模型：Qwen-Plus

数据处理：
- 视频抽帧：ffmpeg和PyAV
- 数据存储：DuckDB轻量分析 或者 ClikeHouse大规模分析
- 流式处理：Kafka和Flink

后端：
- FastAPI，通过RESTful API和WebSocket实现

前端：
- Vue3.js


## 机审内容范围定义
- 短视频：视频流、图片、文本、语音统一对齐的视频内容实体
- 评论内容：文本、图片、评论树节点上下文（保证只删除违规的上下文，不会误删整个评论）
- 广告内容：视频流、图片、文本、语音统一对齐的广告内容实体 

## Prompt审核分层机制
- Prompt L1：直接审核
- Prompt L2：违规风险分类、输出分数
- Prompt L3：业务规则（法律敏感内容等）

## 真实数据来源
- 抖音短视频（凌晨3点-6点、关键词搜索的视频，关键词："审核员睡着了"等）
- 开源抖音APP自带的短视频及评论

为什么要选择3点-6点的短视频？因为3点-6点是大多审核员的休息时段，单靠已部署的机审模型并不足以支撑平台审核工作。

证据实例1：表达隐晦的隐形色情广告
在抖音平台存活的时间：大于3个月（发布时间为2025年4月26日，现为8月）
解决方案：视觉、文字、语音模型分离且并行工作，对所有的

证据实例2：深夜蛋糕
在抖音平台存活的时间：5小时未被处理，在此记录之前的几天早已发现内容相同的视频
出现的问题：视觉模型A/B组全部失效，反复标记且基于人类反馈提示，但模型仍然判定为低风险。
解决方案：强化提示词、审核员复审

证据实例3：旺仔小乔
在抖音平台存活的时间：1小时内未被处理
解决方案：更换模型（Qwen-VL-Plus（A组）, QVQ-Plus（B组），B组没有发现异常，但是A组发现了涉黄问题）

证据实例4：小孩戏耍视频
在抖音平台存活的时间：


## 其他实验数据来源
- NudeNet数据集（主要是图片）
- OpenNSFW2（普通训练集）
- AVA数据集（黄暴内容）
- COCO+任务检测模型（非违规，但是可做正常人像对照集）
- YouTube-8M（擦边视频）


## 过滤标准
- 文本：敏感词、涉政/色情/暴恐、违禁文案
- 图像/视频：低俗画面、暴力、广告、水印、封面与内容不符、格式错误等
- AI生成：未标注AI生成内容、造谣、不实信息、虚拟人未实名注册
- 涉政敏感内容：历史事件/民族冲突/宗教敏感话题等
- 直播内容：性暗示、不健康表达、迷信、审美扭曲、未成年人不适内容

## 通用提示词工程
- 视觉识别模型提示词
  （链接）

- 语音识别模型提示词：
  （链接）


## 文件输出
本项目的左拉菜单有一处能够协助相关工作人员自动生成如下各类文档：
- 策略文档（审核规范，风险分类，判定阈值等）
- A/B测试报告（模型对比，统计分析，效果评估等）
- Prompt实验报告（提示词优化，参数调优，质量评估等）
- 性能对比报告（延迟测试、吞吐量、资源消耗等）
- 数据周报（效能指标，趋势监控，KPI仪表板信息等）
- PRD文档（需求规格，接口设计，用户场景等）

![智能文档生成]()

这是文档生成中的样子，有进度条
![智能文档生成]()

## 文件输出的技术选型
RAG



## 成本控制
假设整个工作流程只调用Qwen模型API，则一个视频内容审核平均消耗8.6k个token，每秒用了1639.7个input token及94.2个output token，按照0.0015元/千 input token和0.0045元/千 output token来计算。假设每天抖音有7800万个新发布视频，抖音官方推荐一个视频播放时间在15秒-3分钟以内，每个视频播放时间取平均值（180+15再取均值）为97.5秒。

* 平均时长: 97.5 s
* input: 1639 tokens/s × 97.5 s = 159,802.5 tokens → 159.8025k × 0.0015 RMB = 0.23970375 RMB
* output: 94 tokens/s × 97.5 s = 9,165 tokens → 9.165k × 0.0045 RMB = 0.0412425 RMB
* 合计: 0.23970375 + 0.0412425 = 0.28094625 RMB/视频
* 全量日成本（7,800 万视频/天）
* 0.28094625 × 78,000,000 ≈ 21,913,807.5 RMB/天
* 等价汇总（按token总量核算）
* 日 input 总量: 159,802.5 × 78,000,000 ≈ 12,464,595,000,000 tokens → 12,464,595,000 × 0.0015 ≈ 18,696,892.5 RMB
* 日 output 总量: 9,165 × 78,000,000 ≈ 714,870,000,000 tokens → 714,870,000 × 0.0045 ≈ 3,216,915 RMB
* 合计: 18,696,892.5 + 3,216,915 ≈ 21,913,807.5 RMB/天

得出结论：每日审核7800万视频，用此套系统开销约2200万元。若本地部署大模型，并引入级联/抽样、小比例重模型概念可将开销进一步压缩。


## 隐私保护措施
- 对所有收集的测试样本作者隐私信息（发表者ID等）进行模糊处理
- 项目遵守MIT协议

## 开源参与者
edwin99（Chen-Jieteng）


