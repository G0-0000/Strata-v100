# 以 VMA 命名标记取代 sidecar 地址登记契约

RAM 七类监控之初版以「主程序登记分配地址区间入旁车文件（ram-pools.txt，STRATA-RAM-POOLS 2 契约），外部读端按区间反推」为标记机制，2026-10-16 部署时仅接通七类中之二类，且须 PID/boot ID/generation/文件锁/mutation 守卫一整套防竞态机制仍不可闭合。兹定：废弃 sidecar 契约及其全套失效机制，改以**内核 VMA 命名标记**（`prctl(PR_SET_VMA_ANON_NAME)`，内核 6.8 实测支持）为唯一定类手段，引擎杂项与「其他」以残差闭合，节点归属以 numa_maps 实测。

## Considered Options

- **扩容 sidecar 契约**（续走旧路补接余下五类）：每接一类须侵入一条分配路径，且 glibc 堆段区间登记天然脆弱（地址复用、VMA 合并），失效协议已证繁复。弃。
- **堆段子区间命名**：命名会分裂 VMA，堆块复用则名留陈旧地址。弃；故二级缓存之段分配须改为自有 mmap。

## Consequences

- 二级缓存（ConversationBuffer 段及存档点成员）分配改为自有 mmap 并命名，现有 NUMA 绑定作用域保留。
- 读端重写为 `maps`+`smaps`+`numa_maps` 快照解析器（本内核 smaps 无 Name 字段，名在 maps 标题行）。
- 清场时 `expert_source.cpp` 内 resident 预算 clamp 修复与登记代码同文件混居，须逐 hunk 撤登记、clamp 秋毫无犯，禁整文件恢复。
- 旧契约文件、守护机制及其测试尽数退役；`strata-memory-monitor.service` 与 `ram-pool-map.conf` drop-in 停删。
