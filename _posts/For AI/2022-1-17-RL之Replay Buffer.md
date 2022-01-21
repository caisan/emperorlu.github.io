# XX:  Efficient Object Storage Service for Reinforcement Learning Applications

> 时间告诉我 没有时间了 —— 卡子

- 或者Decision AI Applications

- 目标：FAST (7月-9月)，~~SC (3月25日)~~，HPCA(7月)，ASPLOS(8月)，ISCA(11月)；~~ATC(1月)，OSDI (1月)，SOSP(奇数年)， DAC(12月)， MICRO(不相干)~~
- 当前困境：项目模式很难搞论文，和他们合作有点过于墨迹
- 当前进展：调研问题，测试瓶颈，寻找创新
  - 我们有环境有机器，用他们的平台直接自己测试，寻找瓶颈

## Abstract

- 加速强化学习的统一对象存储服务 Di-store:  <u>D</u>ecision <u>I</u>ntelligence <u>Store</u>



## Introduce



## Background and Motivation

- RL介绍和训练框架
- Replay Buffer
- 相关问题

### RL和相关训练框架

- OpenAI的Baselines、SpinningUp
- 加州伯克利大学的开源分布式强化学习框架RLlib、rlpyt 、rlkit 、Garage
- 谷歌Deepmind的Dopamine、B-suite 
- 其他独立开发的平台Stable-Baselines、keras-rl、PyTorch-DRL、TensorForce、Tianshou、Di-engine

- 优化方式：multiple/parallel  actors-learners，asynchronous actors-learners

### Replay Buffer

- 经验回放使得在线强化学习的agent能够记住和重新利用过去的经验，在以往的研究中，过去的经验（transition，经验池中的一条记录，表示为元组形式，包含state，action，reward，discount factor，next state）

- 关键问题：一是选择哪些经验进行存储，二是如何进行回放

- Prioritized Experience Replay：回放较为重要的经验，加快收敛

- Distributed Prioritized Experience Replay 

- Parallel Actors and Learners: A Framework for Generating Scalable RL Implementations


### 问题

- 性能瓶颈
  - 随着加速器硬件和高速网络发展，瓶颈在向存储转移 — **需要测试**
  - 目前都是并行训练 / 分布式训练架构，存储瓶颈更明显  — **需要测试**
- 容量瓶颈
  - 随着环境愈发复杂和算法的发展，模型/经验存储容量需求愈高
  - 不同环境 / 算法replay buffer大小  — **需要测试**
  - 存储资源利用率低  — **需要测试**

### 测试

- 基于A100 Ray训练测试



- Replay buffer大小测试



## Design

- 对标：ray-内存object store，NoPFS-预取加速IO，CacheLib，...

- 性能：高性能存储服务，分布式架构（一致性和容错支持），支持高性能网络、共享内存、prefetch...
- 容量：统一资源管理（CPU-memory，NVMe ssd...），大容量，提供offload机制...

## Evaluation



## 计划安排

过年—2月12日：问题调研，学习ray，测试问题

2月13—：待续



## Others

大模型

- 训练效率问题：暂不考虑
- 内存限制问题
- **checkpoint效率问题**

buffer问题：

- buffer大小是策略一开始就确定的嘛 还是会动态扩增
- buffer大小和什么有关？
- buffer大小对性能、精度的影响