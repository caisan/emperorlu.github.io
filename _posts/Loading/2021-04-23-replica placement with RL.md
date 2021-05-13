---
title: 'Replica Placement'
tags:
  - Load_Balancing
---

# Replica Placement 

> **不要温顺地走入那个良宵, 老年应当在日暮时分燃烧咆哮。— 迪伦·托马斯**



[TOC]

## 背景

- 问题（Ceph-Crush 问题）
  - 问题1: 不可控迁移, 迁移量是理论下限的h倍（h为分层结构的层数）
  - 问题2: 不够均衡, CRUSH输入的样本容量不够、副本机制的缺陷
  - 问题3: 选择存储节点时, 仅以节点存储容量为唯一选择条件, 并没有考虑其他因素（到网络和节点
    的负载状况等等）

- 方案: 做两套
  - 靠谱的: 真正可控可靠均匀的机制, 额外代价
    - Crush算法优化
    - balancer、upmap研究和优化
  - 不靠谱的: 发论文, 先进算法, 恐有奇效
    - 算法: 分布算法（见[load balancing.md](2021-04-12-数据分布和负载均衡.md)）, 策略调度（图分割、ML、RL）
    - Replica Placement with RL

## RL简介

- **Reinforcement Learning: An Introduction**, 用于解决基于MDPs的序贯决策问题。[OpenAI](https://spinningup.openai.com/en/latest/index.html)

- 强化学习的设置由两部分组成, 一个是智能体（agent）, 另一个是环境（environment）, 智能体在每一步的交互中, 都会获得对于所处环境**状态**的观察（有可能只是一部分）, 然后决定下一步要执行的动作。环境会因为智能体对它的**动作**而改变, 也可能自己改变。智能体也会从环境中感知到**奖励**信号, 一个表明当前状态好坏的数字。智能体的目标是最大化累计奖励, 也就是**回报**。强化学习就是智能体通过学习来完成目标的方法。

  

  <img src="../../photos/image-20210422193919743.png" alt="image-20210422193919743" style="zoom: 50%;" />

- 基本要素

  - 状态空间（State）: 环境返回的当前情况
  - 动作空间（Action）: 智能体根据当前状态决定下一步动作的策略
  - 奖赏（Reward）: 环境的即时返回值, 以评估智能体的上一个动作
  - 策略、运动轨迹、值函数、Advantage Functions等等

- 发展, 原文总结

  - **ing**
  - 基于Value
    - On-Policy
    - Off-Policy
  - 基于Policy

  <img src="..\..\photos\RL发展.png" alt="RL发展" style="zoom:67%;" />

- 分类, OpenAI的总结

  <img src="../../photos/rl_algorithms.svg" alt="image-20210422193919743" style="zoom: 50%;" />

  - 用的最广的还是Q-Learning、Sarsa、DQN
  
- Q-Learning

  - Q-learning 是一种记录行为值 (Q value) 的方法, 每种在一定状态的行为都会有一个值 Q(s, a)

  - Q table是一种记录状态-行为值 (Q value) 的表

    ![q_learning](..\..\photos\q_learning.png)

  - Q value的更新是根据贝尔曼方程

- Sarsa

- DQN

  - **ing**

## 系统结构领域应用

### Park: An Open Platform for Learning-Augmented Computer Systems

- Tim Kraska, NeurIPS 2019, https://github.com/park-project/park

- **ing**

  
  
- 12个计算机系统中应用场景

  | Environment                     | env_id                         | Committers                       |
  | ------------------------------- | ------------------------------ | -------------------------------- |
  | Adaptive video streaming        | abr, abr_sim                   | Hongzi Mao, Akshay Narayan       |
  | Spark cluster job scheduling    | spark, spark_sim               | Hongzi Mao, Malte Schwarzkopf    |
  | SQL database query optimization | query_optimizer                | Parimarjan Negi                  |
  | Network congestion control      | congestion_control             | Akshay Narayan, Frank Cangialosi |
  | Network active queue management | aqm                            | Mehrdad Khani, Songtao He        |
  | Tensorflow device placement     | tf_placement, tf_placement_sim | Ravichandra Addanki              |
  | Circuit design                  | circuit_design                 | Hanrui Wang, Jiacheng Yang       |
  | CDN memory caching              | cache                          | Haonan Wang, Wei-Hung Weng       |
  | Multi-dim database indexing     | multi_dim_index                | Vikram Nathan                    |
  | Account region assignment       | region_assignment              | Ryan Marcus                      |
  | Server load balancing           | load_balance                   | Hongzi Mao                       |
  | Switch scheduling               | switch_scheduling              | Ravichandra Addanki, Hongzi Mao  |
  
- Server load balancing

  - 任务分配给不同服务器, 使得负载均衡, 并且每个任务运行时间短
  - S: 当前服务器的负载, 进来的任务的大小
  - A: 分配给任务的服务器ID
  - R: 每个任务运行时间的惩罚
  - 每步时间: 1ms
  - 挑战: **1**）输入驱动的变化(input-driven variance), 2）状态表征 (state representation), 3）动作表征 (action representation), **4**）无限长的时间范围 (infinite horizon), 5）仿真现实差距 (simulation reality gap), 6）交互时间慢 (slow interaction time), 7）稀疏空间探索 (sparse space for exploration), **8**）安全探索 (safe exploration)

### 1. 数据库

- 很多过程都可以使用机器学习或者强化学习算法

<img src="..\..\photos\database.png" alt="database" style="zoom: 25%;" />

### 2. 集群调度

Device Placement Optimization with Reinforcement Learning  

3. 组合优化-NP难问题
4. 芯片设计
5. 增强数据, 优化机器学习

## Replica Placement

### An adaptive replica placement approach for distributed key‐value stores

- Concurrency and Computation: Practice and Experience, C类期刊
- 问题抽象

<img src="../../photos/image-20210422193516963.png" alt="image-20210422193516963" style="zoom:67%;" />

- 方案建模

<img src="C:\Users\lukai1\AppData\Roaming\Typora\typora-user-images\image-20210426143706643.png" alt="image-20210426143706643" style="zoom:50%;" />          <img src="C:\Users\lukai1\AppData\Roaming\Typora\typora-user-images\image-20210426143809931.png" alt="image-20210426143809931" style="zoom:50%;" />

- 要求

  - 第一个要求是它必须健壮地适应不同的工作负载模式
  - 我们的模型必须调整自身以适应当前的硬件设置

- 强化学习模型

  - 状态空间

    <img src="C:\Users\lukai1\AppData\Roaming\Typora\typora-user-images\image-20210426144130209.png" alt="image-20210426144130209" style="zoom:50%;" />

    - SNnget: 当前存储节点n中负载, 将v和s的建立映射表, SNnget等于横向之和

      <img src="C:\Users\lukai1\AppData\Roaming\Typora\typora-user-images\image-20210426144326725.png" alt="image-20210426144326725" style="zoom:50%;" />

    - SNnlatency: 是存储节点n响应一个数据请求所需的平均时间

    - MigrationLoad是一个整数, 表示针对R的Get请求的总和, 其中R是一组要迁移的副本

  - 动作空间

    - 操作集合A由存储节点的索引j2n表示, 当前的MigrationLoad将被分配给该节点

  - 奖励定义

    <img src="C:\Users\lukai1\AppData\Roaming\Typora\typora-user-images\image-20210426150342017.png" alt="image-20210426150342017" style="zoom:50%;" />

### 问题抽象

- 副本放置问题: M个数据存到到N个机器上, 每个数据R个副本在不同机器上, 下图中R=2

  ​	<img src="..\..\photos\load.png" alt="load" style="zoom: 25%;" />

  - 目标: 1. 每个机器上的数据尽量均匀
  - 后续保证: 2. 每个机器上的主副本尽量均匀；3. 机器异构环境

- 具体到Ceph

  <img src="..\..\photos\背景ceph.png" alt="load" style="zoom: 25%;" />





### 强化学习建模

- **ing**
- 如何建模, 选择模型, 训练加速, 结果调优, 挑战解决
- 组件: 适用各种数据分布场景 /  针对Ceph特定场景
- 基本





## 参考文献

1. Reinforcement Learning: An Introduction
2. Park: An Open Platform for Learning-Augmented Computer Systems
3. 



 A ControlTheoretic Approach for Dynamic Adaptive Video Streaming over HTTP

Neural adaptive video streaming with pensieve. 

Oboe: auto-tuning video abr algorithms to network conditions



RL在存储系统中的研究

- 问题和挑战
  - 输入驱动的变化 (input-driven variance)
  - 状态表征 (state representation)
  - 动作表征 (action representation)
  - 无限长的时间范围 (infinite horizon)
  - 仿真现实差距 (simulation reality gap)
  - 交互时间慢 (slow interaction time)
  - 稀疏空间探索 (sparse space for exploration)
  - 安全探索 (safe exploration)

- RL for sys 框架平台: **Park(NeurIPS'19)**, 12个相关应用
  - 自适应视频流 (Adaptive video streaming)
    - Neural adaptive video streaming with pensieve, SIGCOMM’17
    - Oboe, ACM Special Interest Group on Data Communication'18
  - Spark集群任务调度 (Spark cluster job scheduling)
    - Resource management with deep reinforcement learning, HotNets'16
    - Auto, ACM Special Interest Group on Data Communication'18
    - Learning scheduling algorithms for data processing clusters,18
  - SQL数据查询优化 (SQL database query optimization)
    - Learning to optimize join queries with deep reinforcement learning, 18
    - Learning state representations for query optimization with deep reinforcement learning, 18
    - A learned query optimizer, 19
  - 网络拥塞控制 (Network congestion control)
    - A hierarchical framework of cloud resource allocation and power management using deep reinforcement learning, ICDCS'17
  - 网络主动队列管理 (Network active queue management)
  - Tensorflow设备放置 (Tensorflow device placement)
    - Device placement optimization with reinforcement learning, ICML'17 
    - A hierarchical model for device placement, International Conference on Learning Representations'18
    - Spotlight: Optimizing device placement for training deep neural networks, ICML'18
    - Placeto: Efficient progressive device placement optimization, NIPS'18
  - 电路设计 (Circuit design)
    - Transferable automatic transistor sizing with graph neural networks and reinforcement learning, 19
    - Learning to design circuits, 19
  - CDN内存缓存 (CDN memory caching)
  - 多维数据库索引 (Multi-dim database indexing)
  - 账户地区分配 (Account region assignment)
  - 服务器负载均衡 (Server load balancing)
  - 交换器调度 (Switch scheduling)
    - Heavy traffic queue length behavior in a switch under the maxweight algorithm. Stochastic Systems'16
- 数据库系统调参
  - Auto-Tuning, ATC'18; Online Reconfiguration, SOPHIA (ATC'19)
  - Heterogeneous Configuration Optimization, Selecta (ATC'18); OPTIMUSCLOUD (ATC'20)
  - **Database Tuning System, SIGMOD'19**
- 分布式系统
  - Data Processing Clusters, SIGCOMM'19
  - Device Placement Optimization, ICML'17
  - **Replica Placement in KV stores**, Concurrency and Computation: Practice and Experience'20
- Cache Replacement 
  - LeCaR, HotStorage'18
  - **Cacheus, Fast'21**
  - Deep Reinforcement Learning-Based Cache Replacement Policy, 20
  - An Imitation Learning Approach for Cache Replacement, ICML'20
- 组合优化-NP难问题; 芯片设计: Chip Design, ISSCC'20
- For us
  - 重点：Park, Tuning, Replica Placement, Cacheus....
  - 场景：Replica Placement ?，Others: Cache ?  Auto-tuning ?





> **怒斥, 怒斥那光的消逝。— 迪伦·托马斯**