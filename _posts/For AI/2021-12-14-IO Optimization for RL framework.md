# IO Optimization for RL Framework

> 那种勃勃生机、万物竞发的境界，犹在眼前 —— 老石

- 目标：A
- 时间：待定
- 问题：待定

## Background and Motivation

### Reinforcement Learning

- 强化学习及其特点

### Framework: from Ray to DI-engine

- 训练框架

### Motivating Observations: IO problems 

- 强化学习训练IO流程梳理，瓶颈？
- 性能优化
  - 算法，通信，IO，预取，并行...
- 内存不足
  - 模型裁剪/优化、超频、算法优化...
  - 单机：卸载到高性能介质/CPU
  - 分布式：Disaggregated Memory （内存分解/内存解聚）

## Related Work

### 1. Memory Wall for Extreme Scale Deep Learning

- [SC '21] ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning
- [ATC '21] ZeRO-Offload: Democratizing Billion-Scale Model Training
  - ZeRO-Offload enables large model training by offloading data and compute to CPU  
- [SC '20] ZeRO: Memory Optimizations toward Training Trillion Parameter Models  
- [FAST '21] Behemoth: A Flash-centric Training Accelerator for Extreme-scale DNNs
- [FAST '21] FlashNeuron: SSD-Enabled Large-Batch Training of Very Deep Neural Networks  

#### 论文1-1. ZeRO-infinity: breaking the GPU memory wall for extreme scale deep learning

- SC '21 Best Paper，microsoft，[DeepSpeed](https://github.com/microsoft/DeepSpeed)开源库：训练1000亿个参数模型的能力

- 背景

  - 在过去的三年里，最大的密集深度学习模型增长了1000多倍，从1亿个参数(ELMo)到超过1000亿个参数(GPT-3)。而GPU内存只增长了5倍(16gb到80gb)。因此，the growth in model scale has been supported primarily though system innovations that allow large models to fit in the aggregate GPU memory of multiple GPUs  
  - 并行技术来训练大模型：model parallelism, pipeline parallelism, and ZeRO  
  - 目前大型模型训练技术中最先进的是3D并行(3D parallelism )，它将模型(张量切片)、管道并行与数据并行相结合，在成百上千个gpu上高效地将DL训练扩展到数万亿参数
  - GPU memory wall：It requires 800 NVIDIA V100 GPUs just to fit a trillion（万亿） parameter model for training  
  
- 深度学习模型训练所需的内存可以分为以下两个部分：

  - 模型状态(OGP)，包括O:优化器状态（例如Aadam优化器中的的momemtum、variance）; G:梯度; P:参数
  - 剩余状态，主要指激活内存(activation memory )
  - GPU Working Memory（临时缓冲区等）
    - Model State Working Memory (MSWM)  ：是模型中最大的单个运算符执行正向或反向传播所需的GPU内存的最小数量。这大约是由模型中该操作符的参数和梯度的大小给出的，因为必须至少有足够的内存来保存参数及其梯度以便向后传播
    - Activation Working Memory (AWM)  ：是反向传播中在执行实际反向传播之前重新计算激活所需的内存，这是两个连续激活检查点之间的激活大小

  - 数字化定量分析：略

    ![image-20211217142948713](..\..\photos\paper\image-20211217142948713.png)

  - Note that it requires 64 GPUs to just fit the model states for a 100B parameter model. Fitting a trillion parameter model requires over 512 GPUs, while a 10 trillion parameter model is beyond the scope of even a massive 1536 GPU cluster  

- 带宽分析

  - 向**CPU内存和NVMe**卸载的一个关键问题是，它们有限的带宽是否会损害训练效率

  - 数字化定量分析：略

  - 结论：不同部分对带宽需求不同，可以在效率和带宽中做权衡

    ![image-20211217142908973](..\..\photos\paper\image-20211217142908973.png)

- ZeRO-Infnity：a novel heterogeneous system technology that leverages GPU, CPU, and NVMe memory to allow for unprecedented model scale on limited resources without requiring model code refactoring  

  - 一种新颖的异构系统技术，它利用GPU、CPU和NVMe内存，在有限的资源上实现前所未有的模型规模，而不需要重构模型代码

  - CPU memory和NVMe构成slow memory，补充GPU容量

  - 划分模型状态，在所有数据并行进程中充分利用聚合内存

    <img src="..\..\photos\paper\SC21-zero.png" alt="SC21-zero" style="zoom: 33%;" />

- Infinity oﬀload engine for model states  

  - ZeRO: Memory Optimizations toward Training Trillion Parameter Models  

    - 现有普遍的数据并行模式下的深度学习训练，每一台机器都需要消耗固定大小的全量内存，这部分内存和并不会随着数据的并行而减小，因而，数据并行模式下机器的内存通常会成为训练的瓶颈

    - ZeRO通过在数据并行进程之间划分OGP模型状态而不是复制它们来消除数据并行进程之间的内存冗余，在训练过程中采用动态通信调度，保持了和数据并行基本一致的计算粒度和通信量，从而保持了计算/通信效率

      <img src="..\..\photos\paper\image-20211217154227520.png" alt="image-20211217154227520" style="zoom: 67%;" />

    - 原来的方式中数据和模型并行都保持了整个训练过程中所需的所有模型状态，但并不是所有的时间都是必需的。例如，仅在某个层的正向传播和反向传播期间才需要与每个层对应的参数。因此，ZeRO通过对参数（包括优化器状态、梯度和参数）进行分区来消除这种内存冗余，**每个GPU仅保存部分参数及相关状态**

    - ZeRO中有三个阶段，对应于三个模型状态:第一阶段(ZeRO-1)只划分优化器状态，第二阶段(ZeRO-2)划分优化器状态和梯度，最后阶段(ZeRO-3)划分所有三个模型状态

  - ZeRO-Infnity建立在ZeRO-3之上，它对所有模型状态进行分区，以消除内存冗余

  - ZeRO-Infinity设计有一个强大的offload机制，称为infinity offioad引擎，它可以将所有分区的模型状态加载到CPU或NVMe内存中，或者根据内存需求将它们保留在GPU上

- CPU Oﬀload for activations：除了模型状态之外，必要时，ZeRO-Infinity还可以将激活内存加载到CPU内存中

- Memory-centric tiling for working memory：为了减少大型模型DL训练对working memory的要求，ZeRO-Infnity引入了一种名为“Memory-centric tiling”的新技术，它利用ZeRO-3的数据获取和释放模式，通过将一个大的操作符分解为可以依次执行的更小的平顶来减少工作内存需求

- 具体设计

  - **ing**

  - 有史以来见过最丑的测试图

    <img src="..\..\photos\paper\image-20211217154747709.png" alt="image-20211217154747709" style="zoom: 50%;" />

  <img src="..\..\photos\paper\image-20211217154723704.png" alt="image-20211217154723704" style="zoom: 33%;" />

#### 论文1-2. Flash-based Memory System for AI

- Behemoth、FlashNeuron(Fast 21)

  

### 2. **Memory Management for Machine Learning**

- [ASPLOS '20] Capuchin: Tensor-based GPU Memory Management for Deep Learning

- [ASPLOS '20] SwapAdvisor: Push Deep Learning Beyond the GPU Memory Limit via Smart Swapping

- [ISCA '19] Interplay between Hardware Prefetcher and Page Eviction Policy in CPU-GPU Unified Virtual Memory

- [ISCA '18] Gist: Efficient Data Encoding for Deep Neural Network Training

- [PPoPP '18] SuperNeurons: Dynamic GPU Memory Management for Training Deep Neural Networks

- [MICRO '16] vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design

  

### 3. **Scheduling & Resource Management**

- [ATC '21] Zico: Efficient GPU Memory Sharing for Concurrent DNN Training
- [NeurIPS '20] Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning
- [OSDI '20] PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications
- [MLSys '20] Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications
- [SOSP '19] Generic Communication Scheduler for Distributed DNN Training Acceleration
- [EuroSys '18] Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters
- [HPCA '18] Applied Machine Learning at Facebook: A Datacenter Infrastructure Perspective



### 4. Memory Disaggregation over RDMA

- [DISAGGREGATION & SERVERLESS](http://wuklab.io/Disaggregation-&-Serverless), UCSD WukLab, Yiying Zhang团队, [github](https://github.com/WukLab)
  - [OSDI '18 best paper] LegoOS: A Disaggregated, Distributed OS for Hardware Resource Disaggregation
  - [ATC '20] Disaggregating Persistent Memory and Controlling Them Remotely: An Exploration of Passive Disaggregated Key-Value Stores
  - [ ASPLOS '22] Clio: A Hardware-Software Co-Designed Disaggregated Memory System
- [ATC '20 best paper] Effectively Prefetching Remote Memory with Leap
- [ATC '21] Zico: Efficient GPU Memory Sharing for Concurrent DNN Training
- wxd...

#### 论文4-1. Disaggregating OS && Disaggregating Persistent Memory

- Distributed Shared Memory (Disaggregated  Memory)，PM，RDMA，多核，混合系统架构...

  ![dsm](..\..\photos\paper\dsm.png)

  - DSM 通过操作系统的内存管理系统把各个独立服务器上的内存地址连接到一起，组成连续的内存地址，使得应用程序可以更方便的做数据共享

  - 随着RDMA的兴起，DSM也随着发展

    <img src="..\..\photos\paper\dsm2.png" alt="dsm" style="zoom: 15%;" />

- splitkernel: hardware resource disaggregation
- LegoOS: 把 CPU、Memory 和 Storage 分别抽象为 pComponent、mComponent 和 sComponent，这些设备之间通过 RDMA 网络连接在一起

![image-20211216141019871](..\..\photos\paper\image-20211216141019871.png)

- Disaggregating Persistent Memory

  - 不分解

    - 在单个节点中，计算和存储之间存在着处理速度方面的差异，无法发挥最佳的性能
    - 可扩展性差
    - 存在数据一致性与可靠性方面的问题

  - aDPM(active disaggregated PM)，主动(active)和被动(passive)是指对数据的管理模式

    - 在aDPM中，将管理程序安装在存储节点，采用这种方式可以降低延迟，但是为了维持较大的网络带宽，在存储节点需要有较高的处理能力，由此会产生较大能耗。此外，如果该系统采用了RDMA技术，那么在这种情况下，需要事先通过管理层才能到达内存，并没有发挥RDMA直达内存的优点

    <img src="..\..\photos\paper\image-20211216112856926.png" alt="image-20211216112856926" style="zoom:50%;" />

  - pDPM(passive disaggregated PM)

    - 在这种模式下，只需要在存储节点安装支持RDMA的智能网卡，就能实现对存储节点内存的直接访问
    - 但在这种模式下，存储节点失去了处理能力，接下来的问题就是在哪里处理与管理数据.从这点出发，提出了三种模式：pDPM-Direct，pDPM-Central和Clover

  - pDPM-Direct：在计算节点进行数据的管理，计算节点通过单向的RDMA对存储节点进行读写操作
    - 对于一条数据，它在存储节点中的形式是一个KV条目，每个KV条目包含已提交和未提交数据，同时这些数据需要有校验码保证可靠性。
      - 当进行读操作时，读取对于KV条目中的已提交数据，并进行校验，如果校验失败，需要重新读取。
      - 当进行写操作时，首先对要写的KV条目加锁，再先后将数据写入未提交和已提交数据中，最后释放锁。
    - 缺点
      - 写操作时较慢
      - 一条数据需要复制为两份保存，会造成空间的浪费

  <img src="..\..\photos\paper\image-20211216113946905.png" alt="image-20211216113946905" style="zoom: 40%;" />

  - pDPM-Central：将数据的处理集中在一个调度器，这个调度器位于计算节点和存储节点之间
    - 在调度器中的PM保存着一张映射表，每个条目保存的是一条数据所在的地址。
      - 当进行读操作时，计算节点会向调度器发送一个RPC请求，调度器会给对应得映射表条目加锁，然后调度器从存储节点读取数据并返回给计算节点，最后释放条目上的锁
      - 当进行写操作时，计算节点会向调度器发送一个RPC请求，此时调度器需要为这条数据在存储节点中分配空间，然后调度器将数据写入分配的空间中，最后更新内部的映射表（需要加锁）
    - 缺点
      - 由于中间经过调度器，读操作的速度下降
      - 调度器本身的CPU使用率非常高，需要处理计算节点的RPC请求、分配存储节点的空间等
      - 调度器成为了该系统的一个瓶颈
  - Clover：对以上两种方式的混合，它将数据和元数据分离，分别采用不同的形式进行管理，其中对于数据的管理（称为数据层），采用的是pDPM-Direct中的方式，即将数据的读写操作分散在每个计算节点中；对于元数据的管理（称为元数据层），采用的是pDPM-Central中的方式，即将数据空间分配和垃圾回收等操作集中在一个元数据服务器(MS)中

### 5. Memory Disaggregation for Deep Learning 

- [Micro ’18] Beyond the memory wall: A case for memory-centric HPC system for deep learning

- [ICAL ‘18] A case for memory-centric HPC system architecture for training deep neural networks

- [TOC] Hierarchical Orchestration of Disaggregated Memory    

  <img src="..\..\photos\paper\image-20211216142622864.png" alt="image-20211216142622864" style="zoom:50%;" />

- [EuroSys '19] Fast Distributed Deep Learning over RDMA

#### 论文5-1. Beyond the memory wall: A case for memory-centric HPC system for deep learning

- 问题：随着DL算法和模型规模的发展，主要受到**评估速度**以及用于训练的**内存大小**的限制

- device-centric deep learning system architecture (DC-DLA)  

  - leading vendors in this space are employing a custom device-side interconnection network that utilizes proprietary high-bandwidth signaling solutions   
  - 设备端：加机器，并行化，用高性能网络，存储设施
  - 问题：问题不在设备短，内存不足memory “capacity” wall  

  <img src="..\..\photos\paper\image-20211216141835383.png" alt="image-20211216141835383" style="zoom: 67%;" />  

- Memory-node architecture  

  - In this paper, we make a case for a memory-centric deep learning system architecture (MC-DLA) that aggregates a pool
    of capacity-optimized memory modules within the deviceside interconnect for transparent memory capacity expansion  
  - 提出了一个以内存为中心的深度学习系统架构(MC-DLA)的案例，该架构在设备端互连中聚集了一个容量优化的内存模块池，用于透明的内存容量扩展  

<img src="..\..\photos\paper\image-20211216142857385.png" alt="image-20211216142857385" style="zoom:50%;" />

### 6. others

- SmartNIC，参考[《SmartNIC Survey》](./2021-12-17-SmartNIC Survey.md)
  - 现有网络带宽逐渐从10GbE增长到100GbE，但同时CPU计算能力增长逐渐缓慢，使得分布式系统的性能瓶颈逐渐从网络转向了CPU。CPU无法提供能够完全利用高速网络带宽的计算能力
  - 因此出现了新的带有计算能力的硬件，这些硬件带有专有的加速器，FPGA或者ARM核心，可以实现将部分CPU的负载进行offload，从而释放更多的CPU资源给应用
  - SmartNIC即为带有计算单元的网卡。根据计算单元的不同通常有三类SmartNIC，分别是流处理器（ASIC），FGPA以及ARM。这三类SmartNIC性能逐渐下降，但是可定制能力逐渐上升。其中基于ARM的SmartNIC可以直接运行完整的Linux并且支持基于C的编程
  - SOSP '21 单独一个主题**Smart NICs**
    - Xenic: SmartNIC-Accelerated Distributed Transactions
    - LineFS: Efficient SmartNIC Offload of a Distributed File System with Pipeline Parallelism （**Best Paper Award**）
    - Automated SmartNIC Offloading Insights for Network Functions

- NVM for ML
  - [Mlsys '19] Bandana: Using non-volatile memory for storing deep learning models



## Design

- 新：RL，架构设计？RL + DSM ？
- 问题：性能？内存？
- 星际争霸trace测试分析
  - 单机下没什么问题
  - 分布式下主要问题也是在传输和数据序列化上
  - 内存不足？是不是CPU/GPU资源太少了，RL训练所需的内存很多？

- 分布式存储资源管理

  - 统一管理，资源分解，大容量，高性能，伸缩性

  <img src="..\..\photos\paper\design.png" alt="design" style="zoom:18%;" />



### Di-store





