---
title:  "Ceph-Seastore"
tags: 
  - Object_Storage
---

本文总结Ceph最新的Seastore。

## 目标 

- Minimize cpu overhead 

- Minimize cycles/iop 

- Minimize cross-core communication 

- Minimize copies 

- Bypass kernel, avoid context switches 

- Enable emerging storage technologies 

- Zoned Namespaces 

- Persistent Memory 

- Fast NVME 

## 方案 

- 使用 SPDK 实现用户态 IO 

- 单进程异步非阻塞模型，Seastar Futures 

- 结合网络模型，一致性协议等，实现零拷贝，来降低在模块中内存拷贝次数 

- 通过 segment 的数据布局来实现的 NMVE 设备的 GC 优化，以及上层如果控制 GC 时的相关处理 