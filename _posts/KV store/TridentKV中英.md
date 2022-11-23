# TridentKV：A Query Optimized  LSM-tree Based KV Store on  NVMe Storage

KV：A Efficient LSM-tree Based KV Store in Read-Intensive Workloads

### ABSTRACT

LSM-tree based key-value (KV) stores 因为多层结构保证高的写性能而遭受严重的读性能损失。当面对更高带宽和低延迟的Modern NVMe solid时，读性能差更凸显无疑。另外由于插入墓碑的删除机制，导致LSM-tree based KV stores存在大规模数据删除导致的读性能波动问题(称为Query-After-Delete)。

为此，我们提出TridentKV，它使用三项关键技术来优化读性能：省空间的分区策略，优化的learned index block结构和结合SPDK的异步设计。TridentKV基于RocksDB实现，我们的测试表明，跟RocksDB相比，不损失写性能的情况下，TridentKV读取性能提升7~12倍，并完美解决数据迁移导致的读性能波动问题。我们也在Ceph使用TridentKV较之前读性能提升20%~60%。

LSM-tree based key-value (KV) stores suffer severe query performance loss because of the multi-layer structure to provide  high write performance. When using modern NVMe solid with higher bandwidth and lower latency, the poor querying perfenormance is undoubtedly more prominent. In addition, due to the deletion pattern of inserting tombstones, LSM-tree based KV stores have the problem of query performance fluctuations caused by large-scale data deletion (Query-After-Delete).

To this end, we propose TridentKV，which improves query performance  via 1)  adopting asynchronous reading design and support SPDK for high concurrency and low latency, 2) designing optimized learned index block structure  to speed up files query, and 3) proposing a space-efficient partition strategy to solve the Query-After-Delete problem.  TridentKV is implemented based on RocksDB and evaluation results show that compared with RocksDB,  TridentKV's query performance is improved by 7x to 12x without loss of write performance and provides stable query performance  without fluctuations even if a large number of deletions or migrations occur.We also use TridentKV in Ceph, and Ceph's  query performance is improved by 20%~60%.



### 1	Introduction

***（KV存储应用）***



以rocksdb为例，它因为优秀的性能和高可用性，常常被用来作为数据库或者分布式系统中的存储引擎，比如MySQL，TiDB，Ceph，Tectonic ，TiDB等等

Persistent key-value (KV) stores are an integral part of storage infrastructure in data centers and have been used in many applications including cloud storage [1], online games [2], advertising [3], [4], e-commerce [5], web indexing [3], [6],and social networks [4], [7], [8].  The state-of-the-art persistent KV stores  such as BigTable [3], Cassandra [13], LevelDB [6], [14], RocksDB [4], and Hbase [15] use a Log-Structured Merge-tree (LSM-tree) [?] improve the performance of write operations by converting small random writes into sequential writes effectively.  Among them, with excellent performance and high availability, RocksDB  is often used as a storage engine in databases or distributed system, such as MySQL[], TiDB[], Ceph[], Tectonic[], etc. 



***(读场景的重要性)***

Read-heavy workloads accounts for a large proportion of current system applications. 

尤其时AI训练场景，读负载占了90%以上。另外，像数据库业务，大多数都是读大于写的，典型的OLTP业务，读写比例可达5：1甚至10：1。tiwtter的memcache 中读写负载比例时7：1，Baidu’s cloud workload also shows读请求时写请求的2.78倍。另外HPC storage systems may no longer be dominated by write I/O - challenging the long- and widelyheld belief that HPC workloads are write-heavy。In fact, over the past few years, read I/O at NERSC has grown to surpass write I/O by a margin, even after accounting for burst-buffer writes! 1.75×

Read-heavy workloads 对所有存储系统能否提供高性能和平稳的查询服务提出了很大的挑战。





So how to optimize read performance is a problem that must be considered when using LSM-tree based KV storage.



For example, 

such as HPC\cite{Revisiting}, AI\cite{AI} and so on.  



数据库，大多数业务对数据库的访问，是读大于写。 典型的如电商、O2O、互联网金融等业务，读写比例可以达到 5:1 甚至 10:1 。

OLTP

AI，训练场景读占了90%以上，

Baidu’s cloud workload also shows a read-write ratio
of 2.78  

tiwtter的memcache 

HPC中，









***（KV存储读性能）***

However,  LSM-tree based KV stores suffer severe query performance loss and read amplification because of log-structured nature of LSM trees, where the query may traverse several levels. 另外由于

在现代存储环境和读密集场景的分析，我们得到两个观察，以此说明



观察1：随着现代设备越来越快，低效的索引问题凸显

Observation 1. 

观察2：删除模式

Observation 2. 











Read-heavy workloads accounts for a large proportion of current system applications, such as HPC\cite{Revisiting}, AI\cite{AI} and so on. 





So how to optimize read performance is a problem that must be considered when using LSM-tree based KV storage.









***（KV存储读性能）***

尽管很多现代KV stores会使用Bloom filter[] 来优化查询，但是读取性能还是短板。另一方面，随着新型存储介质的发展，NVM、3D-XPoint SSD等接口的访问性能比传统hdd提高了2~4个数量级。使用NVMe solid的LSM-tree based KV store 读性能问题更暴露无遗。

LSM-tree can  improve system write performance by converting small random writes into sequential writes effectively.  The files in the storage devices are sorted and stored in multiple levels and merged from lower levels to higher levels.  Because of the multi-layer structure of LSM-tree, the query may traverse each layer，causing serious read amplification and query performance loss. Although many modern KV stores use Bloom filter[] to optimize query performance, which is still a shortcoming. 



如图？？，我们对比了不同线程下使用NVMe的RocksDB性能和NVMe原始性能，

即使使用了128个线程，RocksDB还是不能发挥出NVMe SSD的性能



On the other hand, with the development of new storage media, the access performance of interfaces such as NVM and 3D XPoint SSD has improved by 2 to 4 steps compared with traditional HDD. The poor query performance problem of LSM-tree based KV store using NVMe solid has become more prominent. As show in , we compared the performance of RocksDB using NVMe SSD and the original performance of NVMe SSD under different threads. Evaluation environment and parameters are consistent with that of \ref{eva}. Evaluation results show that RocksDB is far less than NVMe SSD throughput even using 128 threads.  RocksDB cannot give full play to the performance of high-speed  NVMe SSD.

在这篇文章中，我们以Rocksdb为例仔细分析了读过程延迟，发现主要是两部分的问题：1）data access，放盘延迟高，同步读流程限制了并发；2）索引：对于高速存储设备，访盘的延迟越来越低， indexing costs contribute almost equally to the latency。但是Rocksdb的二分查找是不够快的。


In this paper, we carefully analyze the query process delay of RocksDB, which is the representative of LSM-tree based KV store. We find that there are mainly two parts of the bottlenecks: 1) High data access latency. The latency of accessing NVMe SSD is high, and the synchronous IO limits concurrency, and 2) Inefficient files indexing. For high-speed storage devices , indexing costs contribute almost equally to the latency of data accessing[?] and Rocksdb's index search for files is not fast enough.

另外，我们测试发现， 发生大规模删除之后，large-scale deletion，  its query performance was severely reduced. 分析原因，是因为Rocksdb标记删除的删除模式造成的。

In addition，during the use of RocksDB in real world, it was found that when RocksDB undergoes large-scale deletion, its query performance was severely reduced (called Query-After-Delete).  Analyzing the reason, it was found that it was caused by the mark deletion pattern of Rocksdb.



***（提出TridentKV）***

为此，我们提出TridentKV，它采用三种技术来优化读性能：

3. 对于访盘优化，TridentKV使用SPDK减少访问NVMe的延迟，并采用异步设计加速并发。
2. 对于索引优化，TridentKV使用learned index加速查询，并有效的解决string问题；
3. 对于Query-After-Deletion优化，TridentKV设计了一套省空间的分区策略，有效解决大规模删除导致的查询效率低的问题；

我们基于RocksDB实现TridentKV，测试表明跟RocksDB相比，TridentKV读取性能提升7~12倍，使用TridentKV的Ceph较之前读性能提升20%~60%，并且不会有由大规模删除导致的性能波动。

To this end, we propose TridentKV, which uses three techniques to optimize query performance:

\textbf{(1) Asynchronous reading design and SPDK.} For data access optimization, TridentKV uses SPDK to reduce the latency of accessing NVMe SSD and adopts asynchronous design to accelerate concurrency.

\textbf{(2) Learned index block.} For files indexing optimization, TridentKV builds a learned index block to accelerate query and effectively solve the problem of low efficiency of learned index for string keys.

\textbf{(1) Sub Partition.} For Query-After-Delete optimization，TridentKV  designs a space-efficient partition strategy to effectively solve the problem of low query efficiency caused by large-scale deletion.

We implement TridentKV based on RocksDB，and evaluation results show that compared with RocksDB, TridentKV's query performance is improved by 7x to12x. Ceph using TridentKV improves  query performance by 20%~60% compared to using RocksDB.

***（剩余章节）***

本文的其余部分组织如下。第二部分描述了我们工作的背景和动机。我们将在第3节和第4节中介绍TridentKV的设计和实现细节。第五部分介绍了我们的综合实验和评价结果分析。相关工作将在第6节讨论。最后，第七部分是全文的总结。

The rest of this paper is organized as follows. Section 2 describes the background and motivation behind our work. We introduce the design and implementation details of RangeKV in Sections 3 and 4. Section 5 presents our comprehensive experiments and evaluation result analysis. Related work is discussed in Section 6. Finally, Section 7 concludes the paper.

### 2	Background and Motivation

#### LSM-tree and RocksDB



***（Rocksdb介绍）***

RocksDB是一个流行的基于lsm树[17]的KV存储。LSM-tree可以有效地将较小的随机写转换为顺序写，从而提高系统的写性能。如图所示,首先一个LSM-tree批次的写一个叫做memtable固定大小的内存缓冲区。当memtable被填满时，它将被转换成不可变的memtable，然后批处理的数据作为sstable被刷新到存储中。存储设备中的sstable按多级排序存储，并由低一级到高一级合并(如L0, L1…)Ln水平)。电平大小按放大因子指数增长(例如，AF=10)。合并和清理sstable的过程称为压缩。压缩在LSM-tree的整个生命周期内进行，以清除无效/过期的KV项，并在每个级别上保持数据排序，以便有效地读取[18]。如图2（c）所示，每个SSTable都按顺序存储数据，并分为大小一致的块。每个SSTable还具有一个索引块，每个SSTable块具有一个索引条目用于二进制搜索，用于快速检查SSTable中是否存在KV对的Bloom过滤器以及其他元数据（Footer等）。



RocksDB is a popular LSM-tree based KV store. LSM-tree can  improve system write performance by converting small random writes into sequential writes effectively.  As shown in figure ？, firstly an LSM-tree batches the writes in a ﬁxed-size in-memory buffer called memtable . When the memtable is full, it will be converted to an immutable memtable, and the batched data are then ﬂushed to the storage as SSTables. The SSTables in the storage devices are sorted and stored in multiple levels and merged from lower levels to higher levels (e.g. L0, L1..., Ln level). The level size increases exponentially by an ampliﬁcation factor (e.g., AF=10). The process of merging and cleaning SSTables is called compaction. Compaction is conducted throughout the lifetime of an LSM-tree to clean invalid/stale KV items and keep the data sorted on each level for efﬁcient reads[18]. As shown in figure ?， Each SSTable stores KV items in sorted order, divided into uniformly-sized blocks (called data blocks). Each SSTable also has an index block with one index entry per SSTable block for binary search, a Bloom ﬁlter  for quickly checking if a KV pair exists in the SSTable,  and other metadata(Footer et.).

<img src="sec2_RocksDB.png" alt="RocksDB" style="zoom: 33%;" />

读过程

Rocksdb的点查询key K过程可以大致分成八步，其中要注意的是RocksDB支持多种Filter，我们用的是Full类型，即为每个SStable配置一个Bloom filter：

1. MetGet: 首先是在内存中Memtable和Immutable中查找

2. FileFind：如果1中没找到K，RocksDB会在内存中遍历每一层元数据以确定那些SStable可能存在K（称为Candidate SStabe），由于L0层存在overlapping ranges，所以L0层每个SStable都会查找以至找到。其他层是没有key ranges，所以RocksDB通过二分查找找到Candidate SStabes.

   对于每个Candidate Sstable，会先去判断Bloom filter。具体的，如果没在cache中找到Bloom filter，则得

3. FBLoad：将BF Block从磁盘中加载到内存中

4. FBMatch：The filter is queried to check if k is present in the SStable.

   接下来读取Index Block，同样的，如果在cache中没找到，则

5. IBLoad：将Index Block从磁盘中加载到内存中

6. IBSeek：二分查找index block以判断data block的位置

   接下来读取Data Block，如果cache中没找到，则

7. DBLoad：将Data Block从磁盘中加载到内存中
8. DBSeek：二分查找data block以找到K



Lookup a key K in RocksDB can be roughly divided into eight steps. It should be noted that RocksDB supports multiple filters. We use the Full type, which is to configure a Bloom filter for each SStable:

1. MetGet: The first is to search in Memtable and Immutable in memory

2. FileFind: If K is not found in 1, RocksDB will traverse each layer of metadata in memory to determine which SStables may have K (called Candidate SStabe). Since there are overlapping ranges in the L0 layer, each SStable in the L0 layer will  be searched until it is found. The other layers do not have key ranges, so RocksDB finds Candidate SStabes through binary search.

   For each Candidate Sstable, the Bloom filter will be judged first. Specifically, if the Bloom filter is not found in the cache, you have to

3. FBLoad: Load the BF Block from the disk into the memory

4. FBMatch: The filter is queried to check if k is present in the SStable.

   Next, read the Index Block. Similarly, if it is not found in the cache, then

5. IBLoad: Load the Index Block from the disk into the memory

6. IBSeek: Binary search index block to determine the location of the data block

   Next, read the Data Block, if it is not found in the cache, then

7. DBLoad: Load the Data Block from the disk into the memory

8. DBSeek: binary search data block to find K





读延迟分析

我们通过Perf和db_bench等工具对Rocksdb在NVMe SSD上的读过程延迟进行分析，测试环境和参数和。。。一致，我们使用正常大小的cache。图？展示了我们对2中各阶段的延迟占比，其中的缓存了元数据（IB和FB），所以FBLoad和IBLoad延迟为0，没有缓存DB，这是常见的场景。



We  analyze the point query latencies of Rocksdb on NVMe SSD by tools such as perf[?] and db_bench[??]. We use a normal size cache. Figure ?  shows the proportion of  latencies  in each steps in 2, where metadata (IB and FB) is cached, so the delay of FBLoad and IBLoad is 0, and there is no cache DB, which is a common scenario in real systems.



从延迟上分析，主要存在两大瓶颈：

1. 访盘慢，IBLoad占了一半左右的时间，RocksDB并没有发挥出NVMe SSD的性能。很多研究都指出VMe SSD Linux内核的I / O堆栈开销在总I / O延迟中不再可以忽略[?] [?]，对NVMe的使用需要更严谨的设计。另外RocksDB整个读过程是个同步过程，不能利用NVMe SSD高并发的特性。

IBLoad takes about half of the time



2. 索引慢，这个索引主要是指IBSeek和DBSeek，这部分占了15%左右，二分查找性能表现不佳。



In RocksDB using traditional slow devices, files indexing (IBSeek and DBSeek) is performed in memory, which is usually not a performance bottleneck. However, as the devices becomes more and more high-performance, this part can no longer be ignored. As shown in the figure ？ , for NVMe SSD, the latency of IBSeek and DBSeek account for about 15%, which is an obvious performance bottleneck.



Query-After-Deletion



#### 迁移导致的性能下降

***（Scan-After-Deletion问题分析）***

​	下面我们介绍一下Query-After-Migration问题。首先介绍一下RocksDB的Scan和Delete的实现。

Scan and Delete

​	RocksDB基于迭代器实现Scan操作，。RocksDB采用标记删除的策略，删除操作是直接插入数据tombstone。

Scan-After-Delete

在实际应用中，我们发现当RocksDB发生大规模删除之后，其读Scan性能极其低下。当RocksDB发生大量删除的时候，会有很多生成很多tombstone，导致已经删除的key影响其它正常key的查询。



所以在Rocksdb大规模删除是很常见的，包括范围删除和secondary range deletes [?] [?]。RocksDB中，采用标记删除的策略，删除操作是直接插入数据tombstone。当发生大量删除的时候，会有很多生成很多tombstones，导致已经删除的key影响其它正常key的查询。即当发生大规模删除后，RocksDB的读性能尤其是扫描性能会严重下降。

我们以对象存储Ceph为例，它用RocksDB存储元数据，Ceph以PG为单位管理对象，PG是Ceph中最小的数据恢复和数据迁移管理单元。

当Ceph发生数据迁移，一些旧的PG会陆续被删除，导致短时间内，很多PG里面的大量对象都会被删除，同pg的对象在RocksDB中即为scondeary keys，该删除情况就是secondary range deletes。

Query-After-Migration

Next we introduce the Query-After-Delete problem.  An LSM delete operation may be triggered by various logical operations, not limited to user-driven deletes, such as  data migration[?].  Therefore, large-scale delete operations in Rocksdb are very common, including range deletes and secondary range deletes [?] [?].  We take an example with object storage Ceph, which uses RocksDB to store metadata. Ceph manages objects in units of PG (placement group), which is the smallest management unit for data recovery and data migration in Ceph.  As some old PGs will be deleted one after another during the data migration, a large number of objects in many PGs in will be deleted in a short period of time. Objects with the same pg are scondeary (delete)   keys in RocksDB, and the deletion is secondary range deletes.

In practical applications, we found that when RocksDB undergoes  large-scale deletion, the query performance is extremely low. RocksDB adopts the pattern of mark deletion, that is, a delete inserts a tombstone that invalidates older instances of the deleted key. When there are a large number of delete operations, RocksDB will generates a lot of tombstones, which affects query especially scan performance and causes serious space amplification[?].

#### Our Work

***（Our Work）***

我们基于RocksDB实现了具有高效查询性能的TridentKV。

 TridentKV使用异步读取IO并支持SPDK，从而增强了并发性并减少了NVMe SSD访问延迟。英特尔开发的SPDK（存储性能开发套件）[37，69]，一组用户空间库/用于访问高速NVMe设备的工具。 SPDK将驱动程序移至用户空间，避免系统调用并启用零拷贝访问。 它轮询硬件是否完成而不是使用中断，并避免在I / O路径中锁定。

Therefore, we have implemented TridentKV with efficient query performance  based on RocksDB. TridentKV employs asynchronous read IO and supports SPDK, which enhances concurrency and reduces NVMe SSD access latency. SPDK (Storage Performance Development Kit) [？] developed by Intel is a set of user-space  tools and libraries  for accessing high-speed NVMe devices. SPDK  achieves high performance via 1)  moving the necessary drivers into user space，which avoids system calls and enabling zero-copy access,  2）polling hardware for completion instead of using interrupts，and 3）providing lockless resource access.

此外，TridentKV使用学习索引来加快查询速度。[11]首先提出使用learned index来代替传统索引结构加速查找过程，[3]探讨了在LSM-tree结构中使用learned index的可能性和优越性。但是[3]的实现存在很大的缺陷，当learned index预测不对时，其读取代价很大。更糟糕的是Bourbon 对于read-heavy负载表现一般，另外在内存中构建learned index面临丢失的风险，再则Bourbon并未考量对string类型数据的处理。为此TridentKV参考了[8]对bigtable构建learned index的思想，将learned index持久化成sstable中的learned index block，并且设计了一套合理的处理string数据的编码算法。

In addition, TridentKV uses learned index to speed up query. Learned index is first  proposed by [?] to replace the traditional index structure for improving query.  And Bourbon [?]  discussed the possibility and superiority of using learned index in LSM-tree structure. However, the implementation of Bourbon has big flaws: when the learned index prediction is wrong, the querying cost will be very high. To make matters worse, Bourbon still has many problems: 1）Bourbon does not perform well for read-heavy workloads， 2）there is a risk of loss when building learned index in memory，and 3）Bourbon does not consider  the applicability to string keys. For this reason, TridentKV refers to the idea of [8] building learned index for Bigtable[?], persists the learned index into the learned index block in Sstable, and designs a set of reasonable encoding algorithms for processing string keys.

最后，Blue'KV并采用PG分区策略来解决Query-After-Migration问题。

 Last but not least, TridentKV adopts a space-efficient PG partition strategy to solve Query-After-Migration problem.



### 3	TridentKV

#### The TridentKV Architecture

TridentKV整体架构如图5所示，相比RocksDB它使用三项技术来加速读性能：

The overall TridentKV architecture, shown in Figure \ref{TridentKV}, uses three techniques to speed up query performance:  

1. 基于PG分片存储

TridentKV采用省空间的分片存储策略，数据可以按照任意标准，比如key range，secondary (delete) keys等等，分区存储于不同分区结构中（称为PG-tree），每个PG-tree就是一个磁盘里的LSM-tree结构。TridentKV中内存结构和RocksDB一样基于Memtable，但是当flush到磁盘的时候，并不直接将整个Memtable刷下去，而是通过精心设计的PG Scheduler机制将Memtable中kv对按照PG分片存储，分开进行Flush。PG Scheduler包括用于分区的data buffer和加速L0层查找的轻量级L0索引结构。和RocksDB传统列族分区不同，TridentKV所有分区公用了Memtable，减少了内存开销。TridentKV通过将相同PG的数据分布于同一区域，数据迁移直接删除该区域，就完全避免了发生数据迁移后性能停滞的问题，高效快捷。



follow  any  standard

1. PG partition. 

TridentKV adopts partitioned store based on PGs. The data of different PGs are stored in different partition structures (called PG-trees). Each PG-tree is an LSM-tree structure in a disk. The memory structure in TridentKV is based on Memtable and Immutable just like RocksDB. The difference is that when flushing to the disk, instead of flushing the entire Memtable directly, the KV pairs in the Memtable are stored in PG shards through a well-designed PG  Partition Scheduler, and then flushed separately. The PG Partition Scheduler includes data buffers for partitioning and a lightweight L0 index structure to speed up L0 layer lookup. Unlike RocksDB's traditional column family partitioning[?], all partitions of TridentKV share one Memtable, which reduces memory overhead. 

TridentKV distributes the data of one PG in the same area, and data migration directly deletes the area, which completely avoids the problem of performance stagnation after data migration, and is efficient and fast. In addition, partitioned store can effectively reduce the compaction cost because of high spatial locality of real-world workloads [?] [?].  Especially in LSM-based systems, partitioned store can also improve read efficiency and reduce write magnification, because each partition has fewer data and layers.

2. learned index block

TridentKV的每个PG-tree的磁盘结构和RocksDB一样基于SStable，但是摒弃了sstable中index block的设计，使用高效的learned index构建了learned index block加速sstable查找过程。learned index block包括两个模块：String Process和Learned index。 String Process主要负责将string高效的转换成整型数字，用于模型的训练。Learned index我们采用两层的线性模型的rmi模型，它足够处理绝大多数负载[?] [?]。

不同于传统的二分查找，learned index block通过模型直接预测所在的block，相比二分查找速度更快。更加关键的是learned index block相比传统index block小很多，可以在内存中缓存更多，更有助于读取性能。

The disk structure of each PG-tree  is based on SStables like RocksDB. But TridentKV abandons the design of index block in Sstable, and uses efficient learned index to build a learned index block to improve the Sstable query performance. The learned index block includes two modules: String Process and Learned Index. String Process is mainly responsible for efficiently converting strings into integer numbers for model training. For Learned Index, we use a two-layer linear model of RMI [??], which is sufficient to handle most workloads [?] [?].

Different from the traditional binary search, the learned index block directly predicts the block where it is located through the model, which is faster than the binary search. More importantly, the learned index block is much smaller than the traditional index block, and can be cached more in memory, which is more conducive to query performance.



3. 异步读设计并且支持SPDK

如图？？左边所示，TridentKV使用高性能SPDK接口以裸盘的形式来管理NVMe SSD，避免了Linux I/O path带来的系统开销。并且针对读流程设计了异步处理机制，对上下流程提供了异步接口，为减轻竞争I/O资源带来的同步控制开销，以及协调前端和后台处理速度等优化提供了可能。



3.  Asynchronous  reading  design  with  SPDK

As shown in the left of figure [], TridentKV uses the high-performance SPDK interface to manage the NVMe SSD, avoiding the system overhead caused by the Linux I/O path. In addition, an asynchronous IO  is designed for the read process. TridentKV provides  asynchronous interfaces  for the upper and lower processes, which makes it possible to reduce the synchronization control overhead caused by competing I/O resources, and to coordinate the optimization of front-end and background processing speed.



如图？？，一个IO读取过程：首先上层程序调用异步接口访问TridentKV，TridentKV先在内存结构memtable、immutable memtable和PG Partition Scheduler中依次查找。如果未找到，按照PG到对应的PG-tree中查找，找到对应的sstable。接下来是读取sstable，和rocksdb一样，首先读取bloom filter判断是否存在该sstable中。然后通过learned index block快定位其data block，加载到内存并查找目标key。



As shown in figure??  The query process is as follows: the upper-level program calls the asynchronous interface to access TridentKV. TridentKV firstly searches  the Memtable, Immutable memtable and PG Partition Scheduler in turn in the memory. If not found, TridentKV finds in the corresponding PG-tree according to the PG numbe and searchs the Sstable files. For read files, firstly TridentKV read the bloom filter to determine whether it exists. Then quickly locate its data block through the learned index block, load it into the memory and find target key.



#### PG Partition

我们不推荐使用RocksDB列族来实现我们的分区。Ceph中PG数量通常较多，按列族（每个列族维护自己的Memtable）分片会导致Memtable过多，使得内存容量过大。为了缓解这一情况，我们提出一种新的分片机制PG Scheduler，它只需要一个memtable即可完成分区。通过使用data buffer和可配置的轻量级L0索引结构，以block为单位进行flush。额外增加的buffer和index都有容量上限，所以相对于列族或者直接在内存分区的【evendb】方式，内存占用小的多。

We do not recommend using RocksDB Column Family (CF)[??] to implement the partitions. There are usually a large number of PGs in Ceph, so partitioning by CF (each CF maintains one Memtable) will cause too much Memtables and excessive memory usage. In order to alleviate this situation, we propose a new partition mechanism, PG Patition Scheduler, which only requires a Memtable to complete partitioning by a data buffer and a  lightweight L0 index structure. The additional buffer and index have upper capacity limits, so  the memory space is much smaller compared to the CF or the method of directly partitioning in memory [？？].

##### PG Schduler 

如图？？所示，PG Scheduler是以block为单位进行flush，而不是一个sstable。具体的，当flush一个immutable Memtable时，PG Scheduler使用data buffer（数组）先将kv数据按PG分区保存，当某个分区的buffer数据达到一个block的容量大小（默认4K），就将数据排序组织成sstable中block形式（KV有序的），append到L0层。L0中一定数量的block组成一个datable，就类似sstable的data block部分，不包括元数据部分。

As shown in figure ?, PG Partition Scheduler (called PPS) performs flush operations in units of a block, not a Sstable. Specifically, when flushing an Immutable Memtable, PPS uses data buffer (organized into an array) to store KV pairs in PG partitions. When the buffer data of a partition reaches the capacity of a block (default 4K), the KV pairs are sorted and organized  as a block, and appended to the L0 layer. A certain number of blocks in L0 form a Datable, which is similar to the data block part of Sstable, excluding the metadata part.

但是按blcok为单位flush，会导致每个datable中block之间是存在键值覆盖的，查询效率很低
如图？？，PG Scheduler在flush过程中构建内存的轻量级b+tree来加速L0层的查找。类似于SLM-db【？】，但是PG Scheduler只为固定大小的L0层构建索引，所以b+-tree的整体规模不会很大，因此我们只在内存中构建索引，并不持久化，当发生掉电只需重新遍历L0层数据即可恢复，L0层有限的大小使得恢复代价比较小。

However, flush in the unit of blcok will cause KV pairs coverage between blocks in each Datable, and query efficiency is very low. As shown in figure ？ , PPS constructs a lightweight memory  b+tree to speed up the lookup of the L0 layer, similar to SLM-DB [? ]. But PPS only build indexes for the fixed-size L0 layer, so the overall scale of b+-tree will not be very large.

L0层compaction的过程中，将datable合并成RocksDB的sstable形式。和传统的compaction相比，datatable中block之间 key range 事存在重叠的，所以compaction需要以block为单位进行比较和排序，可能会增加文件合并开销，但是都是在内存中进行速度很快。另外datatable只包括data block，compaction造成的写放大会减少。L1层及其之后都是RocksDB sstable的形式，compaction和读取和RocksDB一致。 

In the compaction process of the L0 layer, the Datables are merged into the Sstables. That is, the L1 layer and beyond are all in the form of Sstable, and the compaction and query process are the same as RocksDB.



#### Learned Index Block

我们先对比TridentKV和Bourbon构建模型方法，然后介绍Learned index block和string process

We first compare TridentKV and Bourbon's model building methods, and then introduce Learned index block and string process.

- Learned Index Building

如图？？左边所示，Bourbon[3]构建模型的过程是：1.  读取sstable中每个key的数据集，每条数据是（key，pos）；2.   通过模型训练；3. 训练完成后，通过模型预测key，得到（pos，error），error是误差范围；4. 当预测block中不存在key，则在error范围内所有block中查找。可以看到当模型预测不准时，需要加载很多block到内存中，其搜索代价很大。



As shown on the left of figure\cite{learned}, the model  building process of Bourbon is: 1) Traverse the Sstable to get the data set <K, pos>, pos is the address of the block where the key is located, 2) Train model through the data set , 3)  After the training is completed,  the key is predicted by the model to obtain <pos, error>, and error is the error range，and 4）When the key does not exist in the predicted block, search in all blocks in the error range. Obviously, when the model predicts inaccurately, a lot of blocks need to be loaded into the memory, and the search cost is very high.



我们参考了[8]在bigtable中构建模型的思想：首先通过keys训练learned index，并且预测其所在的block number。然后再将key写入预测的block位置构建sstable。如此通过learned index预测的block必然是正确的，模型预测效果不佳只会导致不同block之间大小不同。具体过程如图？？右边所示：1. 获取文件中每个key的数据集（key，s<key,value>），s表示该key-value对所占字节数；2. 训练模型；3. 构建SStable的过程中，先通过learned index预测一个值S(k)，通过S(k)除以每个block的大小，即得到block number；4.  当

查找过程中通过key预测相应的block num，该block num必定是对的，所以只需加载一次block。

We adopts  the idea of building model in Bigtable from the Google team[?]: first train the learned index through keys and predict the block number. Then write the key to the predicted block location to construct an Sstable. By .writing the data to match the index, the predicted block must be correct, and the error prediction will only cause the size of blocks to be different.  The specific process is shown in the figure? ? (b) : 1) Get the data set {k, s(<k, v>)} of each key in the file. Assume that l(<k, v>) is the number of bytes in record <k, v> and s(<k, v>) denotes the number of bytes preceding k in the SSTable A：
$$
s(\langle{k,v}\rangle) = \sum_{\langle{x,y}\rangle  \in A | x < k}  l(\langle{x,y}\rangle)
$$

$$
\lfloor s(k)/b  \rfloor
$$
2）Train the model，3） During the construction of the SStable,  predict a value S(k) through the learned index, and We can get the block number by $\lfloor S(k)/ \alpha \rfloor$, S(k) is the value predicted by key and $\alpha$ is the block size. Then divide the key into the corresponding data block 4） We store the location of each block location in an array and map a predicted block number to a disk location by . When we get a key, predict the block number by  the learned index and we can get the corresponding block and it must be correct.

- String处理

在现实应用中，key往往是string类型且是变长的。根据[9]的测试，大多learned index处理string都很低效，google 团队的论文[8]也提出了一套转换机制（本文称之为SNBC编码, string number_base coding）. 

In real applications, the key is often the string type and variable. According to the test of Sindex[9], most learned indexes are very inefficient to process strings. The paper of the google team [8] also proposed a set of conversion mechanism (we calls it SNBC, String Number_Base Coding). To quote the introduction of SNBC in the Google team's paper[8], "*we shift and re-base on a character by character basis, based on the range of characters observed at each character position. During training, we determine the minimum and maximum ASCII value of each character position over all keys. We then convert the keys into integers by choosing a different numerical base for each character position according to the range of values that position can take. "* And the paper gives an example, *"consider first how we encode numbers in base 10: 132 = 1 \* 100+3 * 10+2. Now, consider that the only keys are "aab", "bdd", and "bcb". We need 2 values to encode the range of the 0th position, 4 values to encode the range of the 1st, and 3 values to encode the range of the 2nd. Therefore, "bcb" becomes 1 \* (4 \* 3) + 2 \* (3) + 0."

这套编码存在明显缺陷，即当同组的string差距较大时，且越靠前的字符相差越大，转换成的数据会非常大，甚至会超过一个double类型大小上限。我们基于SNBC编码方式，首先将数据进行分组处理，再将其转换成数字。如图？？所示，将keys按照其大小划分几个组（P0到Pm），每个组内keys相差不大，这样再进行SNBC编码会更加高效。

The SNBC coding has obvious flaws, that is, when the string gap in the same group is large, the converted numbers will be very large, and even exceed the upper limit of a double. Based on the SNBC, we first group the data and then convert them into numbers. As shown in figure ? , the keys are divided into several groups (P0 to Pm)  of key rangs. The keys in each group are very close, so that SNBC encoding will be more efficient.

- Model Structure

如图？？，我们选择两层的模型，第一层是一个PLR模型，经过SNBC编码之后，我们得到几个分区的数字，第一层PLR中每个折线段就是一个分区的数据。第二层是多个LR模型。这其实是RMI模型的一个变种，在实际应用中已经能处理绝大多数负载了【】【】。为了保证每个block中key是有序的并且支持scan，模型必须是单调递增的。线性或PLR模型除了简单准确以外，很容易保证单调性。

As shown in figure ? ? , We choose a two-layer model. The first layer is a PLR (Piecewise Linear Regression) model[?]. After SNBC coding, we get the numbers of several partitions. Each polyline segment in the  PLR is the numbers of a partition. The second layer is multiple LR (Linear Regression) models. This is actually a variant of the RMI[?] model, which can handle most workloads in practical applications [] []. In order to ensure that records are still sorted by key and we can still perform range scans, the model must be monotonic.  Because of  simplicity, LR or PLR models are easy to guarantee monotonicity.

- Learned index block

另外传统的learned index都是只存于内存中的，我们将其持久化成sstable中的index block（称之为learned index block）来替代原有的index block形式。Learned index block格式如图？？（b），它和原来的index block（a）最大的区别在于，不需要存储keys，只需要存储模型。图中，block分成两部分，一部分是model parameters，它包括两层的所有参数。另一部分是每个block的BlockHandle，它包括每个block的偏移offset和大小size，在传统index也有，只不过在Learned index block是连续存储的。当然还包括一些元数据信息（校验码之类），这和传统index一样。

The traditional learned index is only the memory structure at risk of loss. We persist it into an index block in Sstable (called the learned index block) to replace the traditional index block. The format of the Learned index block is shown in the figure? ? (B).  The biggest difference between it and the traditional  index block (a) is that it does not need to store keys. In the figure, the block is divided into two parts, one part is model parameters, which includes all the parameters of the two layers. The other part is the BlockHandle of each block, which includes the offset and size of each block, which are also available in traditional indexes, but they are stored continuously in the Learned index block. Of course, it also includes some metadata information (checksums, etc.), which is the same as the traditional index.

我们来简单比较一下两种方式的空间消耗：假设对于1K的KV pairs，64M的SStable大概有65536个KV，假设每个key是16 bytes，那么存储所有keys需要消耗65536 * 16 bytes，即1M左右。对于Learned index block，每个参数是8Bytes，第一个PLR模型，假设是10个折线段，则需要10 * 8 * 2 + 1*8，1代表存储折线段数量（图中Top model num）。第二个模型，假设包括1000个子模型，则需要1000 * 8 * 2 +1*8，1是Sce model num。另外再加上8B的Datablock num，总共只需要不到8K，比传统index缩小了100多倍。



Let's briefly compare the space consumption of the two methods:  Assume that for 1K KV pairs, a 64M SStable has approximately 65536 KVs. Assuming that each key is 16 bytes, it will consume 65536 * 16 bytes to store all keys, which is about 1M. For the Learned index block, each parameter is 8Bytes. For the first PLR model, assuming 10 polyline segments, it needs 10 * 8 * 2 + 1 * 8 bytes, and 1 represents the number of stored polyline segments ("Top model num" in the figure). The second model, assuming that it includes 1000 sub-models, requires 1000 * 8 * 2 +1* 8 bytes, and 1 is the "Sce model num". In addition, with the addition of 8B "Datablock num", it only needs less than 8K in total, which is more than 100 times smaller.



learned index block模型查找速度相对于传统的二分查找速度有不少提升，且提升效果随着sstable的数据量的增加而更加明显。此外由于都是线性模型组成，比传统的index block小很多，空间占用小，cache可以缓存更多的block，总体查找性能可以提升不少。

The query speed of the learned index block has been improved a lot compared with the traditional binary, and the improvement effect is more obvious with the increase of the data volume of Sstables. In addition, Learned index is much smaller than traditional index and can be cached more blocks, and the overall query performance can be improved a lot.

#### SPDK and asynchronous read 

前面提到，整个Ceph-OSD的读取采用的是同步的方式，每次IO请求至少需要访问1-2次磁盘（元数据在内存中命中则只需一次读取数据的磁盘IO，否则至少还需要1次访问元数据的磁盘IO），而每次访问磁盘都是采用同步的方式，主要是调用pread。因此每个线程很可能会进行多次上下问切换，对磁盘的利用率主要取决于线程的数量，但一般服务器的CPU核数是有限的。并且每个服务器需要部署多个OSD，增加每个OSD的线程数会进一步增加CPU线程的调度，因此希望采用异步的方式来读取硬盘，充分的利用每一个线程，尽量使用更少的线程来充分发挥出硬盘的性能。

As mentioned earlier, the entire Ceph-OSD read process is in a synchronous manner. Each IO request needs to access the disk at least 1-2 times (metadata hits in the memory only need to access disk once, otherwise need twice). Each disk access is synchronized, mainly by calling *pread*. synchronous IO does not scale because handling concurrent requests requires one thread for each, leading to context switches that degrade performance when the number of in-flight requests is higher than the number of cores. Therefore, we adpots asynchronous IO for utilizing the performance of fast IO devices. 

TridentKV读过程的异步设计的主要方法是对整体请求链路做分层，每层注册回调函数，自底向上逐层做callback回溯，部分请求链路的异步化调用关系。TridentKV的读取过程比较简洁，根据每次主要读过程，主要有4层分化，第一层是读取cache的时候；第二层是读取Bloom filter的；第三层是读取index的时候；第四层是读取data block。我们目前只对Point-query实现了异步改造，scan和其他操作都是原接口，这是后续工作。

The asynchronous design of the TridentKV reading process is to layer the overall request link.  According to each main TridentKV reading process, there are mainly 4 layers of differentiation. The first layer is for reading the cache; the second layer is for reading the Bloom filter; the third layer is for reading the index block; The fourth layer is to read the data block.  We register the *callback* function at each layer. Then do the callback backtracking layer by layer from the bottom to the top, and the asynchronous call relationship of some request links. We currently only implement asynchronous transformation of Point-query, scan and other operations are the original interface, this is the follow-up work.

相比于同步IO，异步IO就像流水线一样，充分利用每一个线程。比如，在同步IO中，必须等线程#1完成对k1查找后才能进行K2的查找，而在异步IO中当完成第一阶段的查找后，由子线程#1-2完成后续阶段，线程#1异步返回，即可启动线程#2对k2进行查找，并分性争强。当然也不同于多线程，多线程需要更多的cpu资源。

Compared with synchronous IO, asynchronous IO process is like a pipeline, making full use of every thread and enhance concurrency. As shown in the figure \, in synchronous IO, it is necessary to wait for thread #1 to complete the search for k1 before searching for K2. In asynchronous IO, when the first phase of search is completed, subthread #1-2 completes the subsequent phases, thread# 1 asynchronous return and thread #2 to search for k2 will be started. Of course, it is also different from multithreading, which requires more cpu resources.

另外在TridentKV上直接使用SPDK并不能改善性能，In addition, with polling-based SPDK I/O, having threads co-exist on the same cores loses the appeal of improving CPU utilization during I/O waits. 

但是经过异步化设计之后，在使用SPDK时，我们使用专门的线程负责轮询，原来的IO线程直接异步返回。

In addition, using SPDK directly on TridentKV does not improve performance. This is because with polling-based SPDK I/O, having threads co-exist on the same cores loses the appeal of improving CPU utilization during I/O waits. But after the asynchronous design, when using SPDK, we use a separate thread for polling, and the original IO thread directly returns asynchronously.  This greatly reduces the delay caused by polling.

#### Implementation

BlueK基于RocksDB实现，成功封装到Ceph中，当然也可以单独作为读性能高效的KV存储使用。TridentKV中所有的代码基于C++实现，机器学习模型基于MKL库实现。TridentKV保留了RocksDB原有的接口，三项技术都设计成插件形式，即元数据读取接口添加了异步实现的选项，添加了新的SPDK访问接口；添加了Learned index block的选项；添加了分区的选项，分区功能可以按照任意条件分区，并不一定要按照PG，比如按照Key ranges.

BlueK is implemented based on RocksDB and is successfully packaged into Ceph. Of course, it can also be used as a KV storage with high query performance. All the codes in TridentKV are implemented based on C++, and the machine learning model is implemented based on the MKL library[?]. TridentKV retains the original interface of RocksDB, and the three technologies are all designed as plug-ins. That is, the metadata reading interface adds the option of asynchronous implementation, and the new SPDK access interface is added. The option of Learned index block is added and the partition is added. The partition function can be partitioned according to any conditions, not necessarily according to PG, such as according to Key ranges.

### Evaluation

在本节中，我们展示我们的

In this section, we present the evaluation results, which demonstrate the advantages of TridentKV. Specifically, we present the results of extensive experiments conducted to answer the following questions:

1) What are the advantages of TridentKV in terms of the performance? (Section 5.2);

2) What are the effects of three techniques  of TridentKV on the system performance? (Section 5.3);

3) What are the advantages of TridentKV in terms of the performance in Ceph? (Section 5.3).

4) How does TridentKV perform on the Query-After-Migration problem? (Section 5.4).

##### Experiment Setup

All the experiments are run on a test machine from Sensetime Research with Intel Xeon Processor (Skylake) 2.40 GHz processor and 54 GB of memory. The kernel version of the test machine is 64-bit Linux 3.10.0, and the operating system in use is Centos 7.  The NVMe drive is  Intel DC NVMe SSD (P4510 2TB), The basic read and write performances of the NVMe drive are 3200 MB/s (sequential read), 637K IOPS  (random read),2000 MB/s (sequential write), and 81.5K IOPS (random write). 

We compare TridentKV with RocksDB, the base of TridentKV’s development. And we compare Ceph using TridentKV (called Ceph-TridentKV) with Ceph using RocksDB (called Ceph-RocksDB). To make a fair comparison, all the  databases take one thread for compaction and one thread for fushes. The size of Memtables, Immutable memtables, and SSTable is as the default configuration in RocksDB.



##### Bench Performance

我们使用与RocksDB一起发布的db_bench评估两个KV存储的读写性能。 插入的总数据量为40 GB。 密钥的大小为16个字节，值的大小从16B到16 KB不等。

We evaluate the read and write performances of the two KV stores using the db_bench released with RocksDB. The overall inserted data volume is 40 GB. The size of the keys is 16 bytes, and the size of the values vary from 16B to 16 KB.



如图？？和？？ 显示了两个KV存储的随机读取（called RR）和顺序读取（called SR）性能。 比较测试结果，我们可以得出以下结论：

Figures ?? and ?? show the random read （called RR） and sequential  read (called SR performances of the two KV stores. Comparing the test results, we can draw the following conclusions:



1. TridentKV can significantly improve the random read performance. 

从图？？可以看出，相比RocksDB, TridentKV的随机读性能提升7x~12X，这主要归功于异步读设计增加并发性和SPDK增加了读取NVMe的性能，和Learned index对性能的提升。

As shown in figure ？，it can be seen that compared to RocksDB, TridentKV's random read performance is improved by 7x~12X, which is mainly due to the asynchironous read design increasing concurrency and SPDK increasing the performance of reading NVMe, and the performance improvement of the Learned index.



2. 另外，TridentKV can improve the sequential  read performance and 

从图？？可以看出，相比RocksDB, TridentKV的顺序读性能提升7x~12X，且随着value越大提升越明显。对SR性能提升不明显主要是因为RocksDB基于迭代器的顺序读实现，读取一个key之后会把该key所在的block读到内存中加速后续key的读取，TridentKV也基于此实现，所以都是内存中读取，TridentKV的learned index对性能有所提升，但是异步设计不能发挥作用。但是当key size大于block之后（4K），TridentKV的异步设计就发挥效果了。



Figure？ shows  that the sequential read performance of TridentKV is improved by 30%~6.5X compared to RocksDB, and the improvement is more obvious as the value is longer. The insignificant improvement in SR performance is mainly because RocksDB is based on the iterator-based sequential read implementation. After reading a key, the block where the key  located is loaded into the memory to speed up the subsequent key reading. TridentKV is also based on this implementation, so all Read in memory, TridentKV's learned index improves performance. So it's all read in memory and TridentKV's learned index improves performance, but asynchronous design can't work.  But when the key size is larger than the block (4K), the asynchronous design of TridentKV is effective.



图？？可以看出，TridentKV的写性能相比RocksDB没什么差别，稍微些许提升，主要是来自SPDK和learned index带来的。

It can be seen in figure ？ that the write performance of TridentKV is no significantly different than RocksDB, with a slight improvement, mainly from SPDK and learned index.



YCSB perfomance

In this section, we verify the performance of each KV store with YCSB benchmark. The YCSB benchmark is an industry standard macro-benchmark suite delivered by Yahoo!.  Table ? list the seven representative workloads. Workload load is the load process of constructing an 40 GB database; Workload-A is composed with 50% reads and 50% updates; Workload-B has 95% reads and 5% updates; Workload-C includes 100% reads; Workload-D has 95% reads and 5% latest keys insert; Workload-E has 95% range queries and 5% keys insert; Workload-F has 50% reads and 50% read-modify-writes.

可以看到除了load和scan没有提升以外，其他的负载性能提升2x-4x。对于Load E，TridentKV不能提升Scan性能主要是因为并没有对Scan接口异步化设计，并且Learned index对scan也没有提升。Load负载和前面写性能测试差不多，都没有明显提升。其他涉及读的负载都提升明显。

For load E, TridentKV cannot improve scan performance mainly because it has not designed the scan interface asynchronously, and the Learned index has not improved scan. Load load is similar to the previous write performance test, and there is no obvious improvement. Other loads involving reading have increased significantly.



我们对比了不同数据量下RocksDB和TridentKV的随机读性能，RocksDB-opt表示设置了参数使得RocksDB优先缓存元数据，就像TridentKV一样。

从图可以看出，RocksDB随机读性能明显随着数据量增大而降低，这主要是数据量的增大使得RocksDB缓存不了全部元数据，而数据缓存对于随机读而言没有什么作用，所以增加了很多访盘操作去读取元数据。RocksDB-opt和TridentKV性能不随数据量增加而有明显变化，主要是优先缓存元数据，使得即使数据量增大缓存大小足够保证元数据全部缓存，最多只需要一次访盘去读取数据。

访盘操作是最耗时的而且数据缓存对于随机读没有作用，RocksDB-opt和TridentKV通过优先缓存元数据的方式减少了访盘次数，增加了随机读性能。

We compared the random read performance of RocksDB and TridentKV under different data volumes. RocksDB-opt indicated that the parameters were set to make RocksDB cache metadata first, just like TridentKV.

It can be seen from the figure ？ that the random read performance of RocksDB obviously decreases as the amount of data increases. This is mainly because the increase in the amount of data makes RocksDB unable to cache all metadata, so disk access operations are increased  to read metadata. The performance of RocksDB-opt and TridentKV does not change significantly as the amount of data increases, mainly because metadata is cached first, so that even if the amount of data increases, the cache size is sufficient to ensure that all metadata is cached, and at most only one disk access  is required to read the data.

Because data access operations are the most time-consuming and data caching has no effect on random reads，RocksDB-opt and TridentKV reduce the number of data accesses to increase random read performance by preferentially caching metadata.







2. The impact of the three improvements

我们探讨三项技术对TridentKV性能带来的影响，由于我们三项技术的实现是插件方式，所以通过控制变量分别测试三项技术对性能的收益倍数，图？？显示三项技术对读写性能的带来收益的百分比。

We discuss the impact of the three technologies on the performance of TridentKV. Since the implementation of three technologies is a plug-in method, we test the performance gains of the three technologies through the control variables. Figure ？ shows the percentage of benefits of the three technologies to read and write performance.  We can draw the following conclusions:

对于各种负载的读，异步带来的收益都最大。具体的，对于small KV-pairs，learned index收益比SPDK大；对于Long KV-pairs，则相反。对于写而言，三项技术都没有带来明显的性能提升，主要是来自learned index和spdk，pg分区会稍微损失写性能。

下面我们通过展示每项技术的性能表现，来对这个结果进行分析。由于分区对性能基本没有影响，我们在性能测试不再展示，而是放到后面。



For reads of various loads, asynchronous brings the greatest benefits.  

small KV-pairs: Asynchrony brings the greatest benefits, followed by learned index, followed by SPDK;

For Long KV-pairs: Asynchrony brings the most benefits, followed by SPDK, and then Learned index.

For reads of various loads, asynchronous brings the greatest benefits. Specifically, for small KV-pairs, the benefit of learned index is greater than SPDK; for Long KV-pairs, the result is the opposite. For writing, none of the three technologies has brought significant performance improvements, mainly from learned index and spdk, and pg partitions will slightly lose write performance.

Next, we will analyze this result by showing the performance of each technology. Since the partition has no effect on performance, we will not show it in the performance test, but put it behind.





learned index

我们仔细分析learned index对于传统二分查找索引优势在哪里，如图？？，我们对不同value大小和不同cache大小进行性能比较，发现learned index主要从两方面提升读取性能：

- 首先从图？可以看出 KV-pairs越小，learned index提升性能越明显，这主要是因为，value越小，block中的kv对数量越多，会降低binary index block查找速度，learned index不受影响，反而随着数据量增大，模型更加精准。
- 从图？，可以看出cache越大，learned index提升性能越明显，这主要是因为learned index block比binary index block更小，cache可以存储更多block，cache越大，这个优势体现的越明显。

We carefully analyze the advantages of learned index over traditional binary search index, as shown in the figure? . We compared the performance of different value sizes and different cache sizes, and found that the learned index mainly improves the read performance from two aspects:

- First from the figure ?,  it can be seen that the smaller the KV-pairs, the more obvious the improved performance of the learned index is. This is mainly because the smaller the value, the more the number of KV pairs in the block, which will reduce the search speed of the binary index block, and the learned index will not be affected. As the amount of data increases, the model becomes more accurate.

- Figure ？ shows  that the larger the cache, the more obvious the improved performance of the learned index is. This is mainly because the learned index block is smaller than the binary index block and the cache can store more blocks. The larger the cache, the more obvious this advantage is. :



异步+SPDK

我们仔细分析异步和spdk对性能提升的方式是什么，我们测试了不同线程下blueKV和RocksDB对性能的影响

结果发现同步性能随着线程的增加而增加。异步之下，使用很少的线程就可以达到很高的性能。但是随着线程比较多（32），性能有所下降。我们在对比了CPU占用率发现了问题，异步设计增加了异步处理线程，增加了并发，也增加了CPU使用率，随着测试线程的增加，CPU达到了饱和，性能瓶颈慢慢转移到了CPU上。



We carefully analyzed the ways that asynchronous and SPDK can improve performance. We tested the impact of TridentKV and RocksDB on random read performance under different threads，as shown in figure ？.



It turns out that the synchronization performance increases with the increase of threads. Under asynchrony, very high performance can be achieved with few threads. But with more threads (>32), the performance drops. 

We found the problem after comparing the CPU occupancy rate. The asynchronous design increased the asynchrony processing thread, increased concurrency, and also increased the CPU usage rate. With the increase of test threads, the CPU reached saturation, and the performance bottleneck slowly shifted to the CPU.





我们测试了使用不同kv的ceph性能。我们使用Rados bench来进行测试，为了公平起见，后端存储选择Bluestore，所有参数都是按照默认参数。我们从两个方面来进行比较：一个是在16K对象大小下，开启多个bench测试客户端以测试极限性能；二是对不同对象大小进行极限性能比较。

- 图？？显示了开启两个bench可客户端，Ceph性能就达到极限了。使用TridentKV的ceph的随机读性能，比使用Rocksdb提升30%~60%，且非常稳定，没有波动；
- 图？显示，对于不同对象大小的极限随机读性能，使用TridentKV的Ceph都比Rocksdb提升50%以上，这都归功于TridentKV优异的读性能。



In this section, we verify the performance of Cephusing TridentKV and RocksDB to show the advantages of TridentKV.  We use the rados\_bench released with Ceph for testing. For the sake of fairness, Bluestore is selected as the back-end storage, and all parameters are in accordance with the default parameters. We compare from two aspects:  1) to run multiple bench clients to test the extreme performance for the 16K object size, and 2)  to compare the extreme performance of different object sizes.

Figure ？ shows that the performance of Ceph has reached its limit by running two bench clients. The random read performance of ceph using TridentKV is 30%~60% higher than that of Rocksdb, and it is very stable without fluctuations. Figure ？ shows that for the extreme random read performance of different object sizes, Ceph using TridentKV is more than 50% higher than Rocksdb, which is due to the excellent read performance of TridentKV.





Query-After-Migration

在本节中我们测试TridentKV在Query-After-Migration问题表现如何。我们对比了TridentKV的分区策略和将PG按照列族分区（CF Partition），另外还对比了不分区的情况。我们通过删除PG的操作来模拟Ceph数据迁移，通过删除不同数量的PG来展示不同迁移规模的情况。我们设置了以下三组测试：

第一组是测试对比迁移前后在不同分区策略下的性能变化，具体的是设置了10个PG，对集群每隔60s进行性能测试，在T0时刻发生数据迁移（删除1个PG的数据）。图？？可以看到可以看到发生数据迁移后，不分区导致性能下降明显，这主要是因为标记删除导致的。两种分区策略性能不受影响，性能反而提升了，这主要是因为数据量减少了。但是列族方式性能提升更多，而TridentKV的分区方式由于额外索引的原因导致性能不如它，但是相差并不大。

第二组测试时，对不同分区策略的空间放大的测试。我们按照论文【】对空间放大的计算方法，统计了在不同迁移规模情况下三种策略的一个空间放大的情况。具体的是设置了10个PG，然后删除不同PG数量，统计删除前后空间占用，计算空间放大。图？？显示，不分区的空间放大随着删除比例增加明显，而其他两种分区策略不会因为数据迁移而造成空间放大，主要是因为迁移是立即删除整个分区，而不是传统的标记删除方式。

第三组，我们对比了不同pg数量三种策略内存使用率情况。具体的，我们依次增加PG数量，观察不同分区策略的内存使用量。如图？？，可以看到随着pg数量增加不分区不受什么影响，而PG分区会增加到某个阈值（约6%）就不再增加，这是因为PG Partition Schedule使用内存有上限。而CF_Partition随着PG数量增加，CF和memtable数量增加，内存使用率明显增加。当然可以将每个列族的memtable大小减少，但是会导致每个sstable很小。

总而言之，TridentKV的PG_Partition可以保证Ceph数据迁移之后性能不损失，且不占用过多内存。

In this section, we test how TridentKV performs in the Query-After-Migration problem. We compared TridentKV's PG partitioning strategy with PG partitioning according to column family (CF Partition), and also compared the case of non-partitioning. We simulate Ceph data migration by deleting PGs, and show different migration scales by deleting different numbers of PGs. 

Comparing the test results, we can draw the following conclusions:

The first group is to compare the performance changes caused by different partitioning strategies before and after the migration. Specifically, 10 PGs are set, and performance statistics are performed on the cluster every 60s. Data migration occurs at T0 (the data of 1 PG is deleted), and T0  is in the middle of time 2 and time 3.  As shown in figure ? , it can be seen that after the data migration occurs, the performance degradation caused by non-partitioning is obvious, which is mainly caused by the mark deletion. The performance of the other two partitioning strategies is not affected, but the performance is improved, mainly because the amount of data is reduced. However, the performance of the column family method has improved more, and the performance of TridentKV's partition is not as good as it due to the additional index, but the difference is not big.



The second group is  to evaluate the space amplification of different partitioning strategies. According to the  space amplification calculation  of Lethe[??], we counted the space amplification of the three strategies under different migration scales. Specifically, we set up 10 PGs, then delete the different  numbers of PGs. We count the space occupied before and after the deletion, and calculate the space amplification. Figure ？shows that the space amplification without partitioning increases significantly with the deletion ratio, while the other two partitioning strategies will not cause space amplificationdue for data migration. This is mainly because in the two  partitioning strategies，the migration deletes the entire partition immediately, rather than the traditional mark deletion method 



In the third group, we compare the memory usage of the three strategies with different pg numbers. Specifically, we sequentially increase the number of PGs and observe the memory usage of different partitioning strategies. As shown? ? , it can be seen that as the number of pgs increases, the non-partitioning  is not affected, and the TridentKV's PG partition will increase to a certain threshold (about 6%) and no longer increase. This is because the memory used by the PG Partition Schedule has an upper limit. And for CF Partition, as the number of PGs increases, the number of CFs and memtables increases, and memory usage increases significantly. Of course, the memtable size of each column family can be reduced, but each sstable will be small.

All in all, TridentKV's PG partition can ensure that performance is not lost after Ceph data migration, and does not occupy too much memory.



1. 读性能

<img src="C:\lukai1\桌面\论文写作\TridentKV\图片\test_1_readrandom.png" alt="test_1_readrandom" style="zoom: 50%;" /><img src="C:\lukai1\桌面\论文写作\TridentKV\图片\test_2_readseq.png" alt="test_2_readseq" style="zoom: 50%;" />

2. 写性能

<img src="C:\lukai1\桌面\论文写作\TridentKV\图片\test3_write.png" alt="test3_write" style="zoom:50%;" />

3. 不同线程的性能表现

   <img src="C:\lukai1\桌面\论文写作\TridentKV\图片\test4_read_thread.png" alt="test4_read_thread" style="zoom:50%;" />

YCSB

<img src="C:\lukai1\桌面\论文写作\TridentKV\图片\test5_ycsb.png" alt="test5_ycsb" style="zoom:50%;" />



<img src="C:\lukai1\桌面\论文写作\TridentKV\图片\test14_gain.png" alt="test14_gain" style="zoom:50%;" />



learned index



<img src="C:\lukai1\桌面\论文写作\TridentKV\图片\test6_learned_cache.png" alt="test6_learned_cache" style="zoom:50%;" /><img src="C:\lukai1\桌面\论文写作\TridentKV\图片\test7_learned_value.png" alt="test7_learned_value" style="zoom:50%;" />





ceph



<img src="C:\lukai1\桌面\论文写作\TridentKV\图片\test13_time.png" alt="test13_time" style="zoom:50%;" /><img src="C:\lukai1\桌面\论文写作\TridentKV\图片\test11_ceph_mem.png" alt="test11_ceph_mem" style="zoom:50%;" />



<img src="C:\lukai1\桌面\论文写作\TridentKV\图片\test12_sa_pg.png" alt="test12_sa_pg" style="zoom:50%;" />

<img src="C:\lukai1\桌面\论文写作\TridentKV\图片\test9_ceph_size.png" alt="test9_ceph_size" style="zoom:50%;" /><img src="C:\lukai1\桌面\论文写作\TridentKV\图片\test8_ceph_bench.png" alt="test8_ceph_bench" style="zoom:50%;" />



[测试](C:\lukai1\桌面\论文写作\TridentKV\测试部分.md)

- 总体性能

  - db_bench

  可以看到对于读性能，TridentKV提升了7.1x-12.7x的随机读性能，，顺序读性能；写性能并没有提升，顺序写性能稍好于RocksDB。

  - YCSB

- 每项测试

  - 分区
    - 删除问题
  - learned index
  - 异步+spdk

- ceph

  -  rados bench

### Related Work

1. kv的优化

基于LSM-Tree的KV存储写放大优化，LSM-Tree进行compaction的时候会将SST从磁盘上读出，在内存中排序之后再写回，这个过程造成了写放大。典型的优化策略包括键值分离和允许部分键值覆盖策略。

KV storage write amplification optimization based on LSM-Tree. When LSM-Tree performs compaction, SST will be read from the disk, sorted in the memory and then written back. This process causes write amplification. Typical optimization strategies include key-value separation and partial key-value coverage strategies.

Wisckey[16]首先提出了键值分离的策略，Wisckey核心思想是将key与value分开存储，value则以日志的形式进行管理，key与对应的value在日志中的位置记录在LSM中，大大缓解了写放大问题。但是这个方案也有一定的限制，一是对于value比较小的情况，KV分离的方案对比普通LSM-Tree结构并没有优势，反而scan性能还会更差；二是KV分离方案的GC仍需要进行谨慎的设计，一方面需要及时GC回收空间，另一方面也需要考虑GC对存储设备IO带宽占用后对前台操作的影响。HashKV [17]主要是针对Wisckey中高额的GC开销问题进行的优化，HashKV通过与key对应的哈希将数据划分成固定大小的分区存储在数据仓库中。这样就可以达到分区隔离和固定分组的效果，使GC变得灵活和轻量，分区的大小是动态调整的，并且允许每个分区通过分配预留空间的方式进行增长。为了提升分区空间的利用效率以及GC效率，HashKV引入了热度识别算法，对冷热数据进行区分，热数据在GC的时候可以避免迁移冷数据，减少数据迁移量。UniKV [18] 采用两层结构，第一层是无序存储，允许 SSTable 键值范围覆盖；第二层是有序存储，保存从无序存储合并而来的 SSTable。无序存储中的键值对通过内存中的哈希表加速查找。有序存储中键值分离，避免合并过程中 value反复迁移造成的写放大。另外 UniKV 动态地增加分区，通过横向扩展避免 LSM-Tree因层数增加带来的写放大。

第二个策略典型代表是PebblesDB[19]，借鉴Skiplist的思想，每层划分range，range内允许键值覆盖。使得每层之间不一定要强有序，从而缓解写放大。LSM-trie[20]用key的哈希值构造前缀树，并且允许覆盖。SifrDB[21]通过全局索引合并，子树组织数据的形式减小因合并操作产生的写放大。这些允许部分键值覆盖的方法，虽然能很好的控制写放大问题，但是在磁盘上的数据组织复杂，进一步牺牲读性能。



另一方面是关注KV存储的读优化研究，由于LSM-tree多层结构的设计，牺牲了一定的读性能，目前的研究主要从三个方向对其进行优化：过滤器优化、索引结构优化和Cache优化。

👌

The read performance of LSM-tree is sacrificed due to the multi-layer structure. The current research mainly optimizes it from three directions: filter, index structure and cache optimization.

 bLSM[22]第一次使用bloom filter来优化读性能，通过filter可以快速判定元素是否存在，而不必实际读取，大大加速了读取效率。但是bloom filter也存在内存占用率偏高、严重的误报率和不支持范围查询等问题。ElasticBF[23]通过细粒度的bloom filter分配，并根据数据的冷热程度动态地调节bloom filter的bloom bits长度，从而达到降低内存占用的目的。Succinct Range Filter(SuRF)[24]实现了一种叫做FST(Fast Succinct Trie)的数据结构，它既拥有高压缩特性, 还可以实现快速的点查询和范围查询。FST本质上是一种高度优化之后的字典树, 其实可以实现静态词典的数据结构。论文中使用FST替换掉了RocksDB的Bloom filter, 支持范围查询，并且在相同存储空间的情况下获得了查询性能的提升。SlimDB[25]使用cuckoo filter 来替代bloom filter，并构建了多级Cuckoo Filter。

我们认为Bloom filter已经足够好了，对filter的优化对性能提升不大。

👌

For LSM-tree query performance optimization, bLSM [22] first uses bloom filter, which can quickly determine whether the kv item exists without actually reading it. However, bloom filter also suffers from high memory usage, serious false positive rates, unsupported range queries and so on.

 ElasticBF [23] adopts a fine-grained heterogeneous Bloom filter management scheme with dynamic adjustment according to data hotness, so as to achieve the purpose of reducing memory usage. 

Succinct Range Filter (SuRF) [24] supports  common range queries with  a fast and compact data structure for approximate membership tests.

 SlimDB [25] uses cuckoo filter to replace bloom filter, and builds multi-level cuckoo filters.







第二点是索引结构的优化，SLM-DB[26]在NVM上实现一个持久性的全局B+树索引，另外将LSM-tree结构改成单层的方式加速读过程。X-Engine [27]构建了多版本元数据索引、优化cache、减少分层根据分层进行冷热分离和硬件FPGA加速compaction等多种措施来优化查找。Bourbon[28]使用机器学习的方法，通过静态SStable数据构建学习型索引来加速Sstable查找。纵观这些研究，要么改变LSM-tree本身多层结构，会很大程度上影响LSM-tree优秀的写性能；要么添加辅助索引，就会增加很多额外的开销，而且这些索引本身也存在很多缺陷。TridentKV不添加额外的内存索引结构，而是将learned index持久化到Sstable中，以此减少内存开销。 

SLM-DB[26] implements a persistent global B+ tree index on NVM, and Kvell[??]  adopts a variety of memory index structures. Bourbon[28] builds learned index through static Sstables data to speed up Sstables search. These studies either change the multi-layer structure of LSM-tree, which will greatly affect the excellent write performance of LSM-tree, or add auxiliary indexes, which will increase a lot of additional memory overhead, and these indexes themselves also have many defects. TridentKV does not add an additional memory index structure, but persists the learned index to Sstable to reduce memory overhead.



 最后是cache的优化，AC-key[29]通过构建多种cache组件，对于不同的负载提出自适应缓存策略。LSbM[30]发现compaction之后会有大量cache失效的问题，其通过增加合并操作的缓存来缓解。缓存确实可以极大提升读取性能，但是不希望通过增加缓存而使得内存不够用。

Ac-Key [29] proposes an adaptive cache strategy for different loads by building a variety of cache components. LSBM [30] found a large number of cache failures after compactions, which were alleviated by increasing the cache for merge operations. The cache can indeed greatly improve the read performance, but we do not want to increase the cache and make the memory insufficient.

2. ceph的优化

有很多研究关注全NVMe闪存下Ceph性能问题，从18年开始，Red Hat就领衔社区针对NVMe闪存重构osd框架，提出Seastore[31]，主要从以下几个方面进行：一是面向NVMe 设计，不考虑 PEME和 HDD；二是使用SPDK 实现用户态 IO；三是通过 segment 的数据布局来实现的 NMVe设备的 GC 优化，以及上层如果控制 GC 时的相关处理；四是使用Seastar框架进行基于单进程异步非阻塞模型实现run-to-completion；五是结合Seastar 的网络消息层实现读写路径上的零（最小）拷贝。但是Seastore大面积的重构，开发周期很长，另外对实际应用而言，进行针对性优化即可。韩国首尔大学的Myoungwon Oh团队[32]在ICoCC’16也提出在全闪存下ceph性能不佳，提出三点优化：细粒度锁、无阻塞日志优化和合适参数匹配。但是其偏向写优化, 大部分针对Filestore，锁机制的优化对于性能提升有限。Intel[13]从硬件和接口方面，基于高性能网络、dpdk和spdk等对ceph进行全方面的优化，对架构本身优化少。SAPPHIRE[33]则从参数调优的角度入手，基于机器学习自动调优，性能提升有限。

有很多研究关注全NVMe闪存下Ceph性能问题，从18年开始，Red Hat就领衔社区针对NVMe闪存重构osd框架，提出Seastore[31]，但是Seastore大面积的重构，开发周期很长，另外对实际应用而言，进行针对性优化即可。韩国首尔大学的Myoungwon Oh团队[32]在ICoCC’16也提出在全闪存下ceph性能不佳，提出三点优化：细粒度锁、无阻塞日志优化和合适参数匹配。但是其偏向写优化, 大部分针对Filestore，锁机制的优化对于性能提升有限。。Intel[13]从硬件和接口方面，基于高性能网络、dpdk和spdk等对ceph进行全方面的优化，对架构本身优化少。SAPPHIRE[33]则从参数调优的角度入手，基于机器学习自动调优，性能提升有限。

这些研究没有关注到Ceph的读性能，也没有指出Rocksdb的问题和Query-After-Migration，TridentKV极大提升了Ceph读取性能并且解决。。。

There are many studies focusing on Ceph performance issues under  NVMe SSD. Since 18 years, Red Hat has led the community to rebuild the osd framework for NVMe SSDs and proposed Seastore [31]. However, Seastore has been refactored in a large area and the development cycle is very long. In addition, For practical applications, targeted optimization is sufficient. Youngwon Oh[32] also proposed that ceph performance is poor under all-flash memory, and proposed three optimizations: fine-grained locking, non-blocking log optimization, and appropriate parameter matching. It is mainly optimized for write optimization, mainly for Filestore, which is already rarely used by people. Intel [13] optimizes  Ceph based on high-performance network, dpdk and spdk in terms of hardware and interface, and optimizes the architecture itself less. SAPPHIRE [33] started from the perspective of parameter tuning, and automatically tuned based on machine learning, with limited performance improvement.

These studies did not pay attention to Ceph's read performance, nor pointed out the Rocksdb problem and Query-After-Migration. TridentKV greatly improves Ceph's read performance and solves  the query performance fluctuation caused by data migration



 针对Ceph性能优化的研究中，有很多文章\[34][35]从网络模块上进行优化，他们发现当连接过载时，它将导致工作线程之间出现负载不平衡的问题。[34]根据工作负载量，工作线程之间做到负载平衡，另外存储服务器之间多个连接，每个单个连接分配多个工作线程。[35]则优化锁机制解决负载竞争。还有一些性能优化研究，[36]基于librados的文件下载/上传文件提出多线程性能优化，[37]采用弱一致性和多节点读取优化存储策略，实现异构SSD环境下读写优化，但是不提供强一致性保障。[38] 在[36]的基础上，使用了多个流水线算法来优化将大文件写入Ceph存储集群的性能。

3. learned index

   由Tim Kraska【】提出的Learned Index Structures近几年研究非常多，它主要观点是索引就是模型，从而使用高效的机器学习模型构建索引结构来替代传统索引，并取得很好的性能提升。但是最初的Learned Index存在只读、单线程、in memory、 不支持string keys等等问题。ALEX、XIndex、PGM-index等等研究为Learned index写性能和高并发提供了解决思路，Sindex致力于支持string类型。也有很多研究将Learned index在实际系统中运用，Xstore将其在基于RDMA的KVs中缓存应用，Bourbon在leveldb中，Google团队在BIgtable中运用。

   TridentKV首次将Learned index运用于Rocksdb，并改进了构建模型的方式，将Learned index持久化成block，并很好的支持string keys。

   

   The Learned Index Structures proposed by Tim Kraska [] has been studied a lot in recent years. Its main point is that the index is a model, so that efficient machine learning models are used to build  index structures to replace the traditional indexes and achieve good performance improvement. However, the original Learned Index has problems such as read-only, single-thread, in memory, and does not support string keys. Researches on ALEX, XIndex, PGM-index, etc. provide solutions for Learned index write performance and high concurrency. Sindex is committed to supporting string keys. There are also many studies that use the Learned index in actual systems. Xstore uses it in RDMA-based KVs for caching, Bourbon in leveldb, and the Google team 【】uses it in BIgtable.

   TridentKV applies the Learned index to Rocksdb for the first time, and improves the way of building the model, persists the Learned index into a block, and supports string keys well.

   

4. 分区

   partitioned store can effectively reduce the compaction cost because of high spatial locality of real-world workloads. Especially in LSM-based systems, partitioned store can reduce the time and space requirement of compaction processes. Rocksdb提供了列族分区的方式，EvenDB在内存中进行分区。

   这些方式都会导致内存过大，Rmix【】的方式对于分区方法描述太简略。

   我们通过使用有容量上限的分区策略，且不局限于按照key range分区，可以按照任意标准。在Ceph中我们按照PG分区，完美解决了Query-After-Migration问题。



Rocksdb provides a way to partition by column families, and EvenDB[?] partitions in memory.
These methods will cause the memory to be too large, and the Rmix [?] is too brief for the description of the partition method.

We use a partition strategy with an upper capacity limit, and are not limited to partitioning according to the key range, and can follow any standard. In Ceph, we follow the PG partition, which perfectly solves the Query-After-Migration problem.



3. 异步

   spandb udopt 

### Conclusion

和RocksDB相比，异步的设计加速并发，SPDK减少延迟；分片策略减少查找范围，读取效率更高，每个分区层数减少，读写放大也有相应减少，最重要的是极大解决了大规模删除后读取效率问题；Learned index加速了SStable的查找。整体而已，相对于RocksDB，TridentKV在不牺牲写性能的情况下极大提升了读取性能，并且完美解决Ceph中RocksDB大规模删除问题。

Compared to RocksDB, the asynchronous design accelerates concurrency; The sharding strategy reduces the search range, higher reading efficiency, the number of each partition layer is reduced, and the read-write amplification is also reduced correspondingly. The most important thing is that it greatly solves the problem of reading efficiency after large-scale deletion. Learned Index speeds up the search of SSTables. Overall, TridentKV greatly improves read performance compared to RocksDB without sacrificing write performance, and perfectly solves the problem of massive RocksDB deletions in Ceph.

#### References

[1] Weil S A, Brandt S A, Miller E L, et al. Ceph: A scalable, high-performance distributed file system[C]//Proceedings of the 7th symposium on Operating systems design and implementation. USENIX Association, 2006: 307-320.

[2] 

[3] Windows azure storage: a highly available cloud storage service with strong consistency

[4] The hadoop distributed file system

[5] Lustre: building a file system for 1000 clusters

[6] The google file system

[7]Human Brain Project 

[8] Gpfs: a shared-disk file system for large computing clusters

[9] F4: Facebook’s warm blob storage system

[10] Tyr: blob storage meets built-in transactions

[11] 基 于 高 性 能 I/O 技 术 的 Memcached 优化研究

[12] bluestore

[13] RocksDB

[14] ceph  intel

[15] rados

[16] crush

[17] P. O’Neil, E. Cheng, D. Gawlick, and E. O’Neil, “The log-structured merge-tree (lsm-tree),” Acta Informatica, vol. 33, no. 4, pp. 351–385, 1996.

[18] R. Sears and R. Ramakrishnan, “blsm: a general purpose log structured merge tree,” in Proceedings of the 2012 ACM SIGMOD International Conference on Management of Data, 2012, pp. 217–228.

evendb

remix

EvenDB【】的方式也会导致内存过大，Rmix【】的方式对于分区方法描述太简略。

最大带宽39.4GB/s，







