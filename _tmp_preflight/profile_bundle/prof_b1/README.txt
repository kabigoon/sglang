collect_profile_bundle.py —— 从 torch_npu profiler 产物里抽出**小体积**证据包，便于 git 推送。

为什么需要它：一个 rank 的 ASCEND_PROFILER_OUTPUT 动辄上百 MB，但真正要看的信息
（按 stream 聚合、按算子聚合、top kernel、每步耗时）加起来只有几百 KB。

用法：
  python3 _tmp_preflight/collect_profile_bundle.py /tmp/prof_b1 --tag prof_b1
  python3 _tmp_preflight/collect_profile_bundle.py /tmp/prof_128k --tag prof_128k --max-kb 400

产物目录（git 可推）：
  _tmp_preflight/profile_bundle/<tag>/
    README.txt            本包是什么、怎么生成的
    meta.txt              机器/时间/git sha/源目录
    stream_summary.csv    每个 Stream ID：kernel 数、总耗时、占比、top 算子  ← 认泳道用
    step_trace_time.csv   每步耗时（原样复制）
    op_statistic_top.csv  算子聚合 top N
    kernel_top.csv        耗时 top N 的 kernel（含 pipe 比率列）
    comm_summary.csv      HCCL 通信汇总
