# ADSTS （27天）





## Abstract





## Introduction



## Background and Motivation



## Design

In this section, a new RL-based auto-tuning  scheme for distributed storage systems will be described. The whole ADSTS framework is introduced first. Then, parameter processing and tuning models are introduced  in detail.

### ADSTS Architecture





<img src="C:\lukai1\桌面\论文写作\自动调参\ICPP-Auto.png" alt="ICPP-Auto" style="zoom: 25%;" />



Fig.\ref{ADSTS} illustrates the architecture of ADSTS, a tuning service that works with any distributed storage system. 



- Parameter Preprocessor 
  - 
- Common Interface
- Tuning Agent
- Memory Pool



## Parameter Model

处理参数的第一步是得到所有参数设定，这不是一件容易的事，因为很多参数调优并没有得到关注，因此缺乏相应地文档。因此针对这些系统，我们会analyze the configuration source code directly ，获取完整的参数列表。

Based on the multiple modules structure of the distributed storage system, ADSTS arrange module selecting parameters into a tree-based indexing structure (\textit{Parameter Tree}) and classify parameters based on which module and sub-module they belong to.  

这是个费事的事情，而且对于不同系统可能会有不同的决策，对于有

前缀列表

entry [name,type,level,desc,default,tags,services,min,max]



get the complete parameters set ，lacks attention from Ceph developers  



Parameter Pruning



Parameter Preprocessing



Parameter Identification







## Tuning Agent






## Optimization



## Evaluation



## Conclusion



