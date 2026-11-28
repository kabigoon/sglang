# vLLM-ascend profiler 摘要（k128k）

- 来源: `/home/a00821909/va_prof/dp2_pp0_tp0_dcp0_ep8_rank0_110056_20261126140227624_ascend_pt/ASCEND_PROFILER_OUTPUT`
- kernel 数: **146304**，总耗时: **29982.7 ms**
- 切步: **65 步**（间隙阈值 2000 us）

## 算子桶

| 桶 | kernel 数 | 总耗时(ms) | 占比 | 平均(us) |
|---|---|---|---|---|
| comm | 22104 | 8208.3 | 27.38% | 371 |
| indexer | 1248 | 5892.6 | 19.65% | 4722 |
| mla_attn | 2064 | 4885.4 | 16.29% | 2367 |
| mhc | 7680 | 3909.4 | 13.04% | 509 |
| gemm | 10743 | 2157.8 | 7.20% | 201 |
| other | 42203 | 1492.5 | 4.98% | 35 |
| norm | 36184 | 1158.1 | 3.86% | 32 |
| moe | 12160 | 1142.0 | 3.81% | 94 |
| quant | 11918 | 1136.6 | 3.79% | 95 |

## 步（chunk）

| # | span(ms) | kernel 数 | 占比 | top 算子 |
|---|---|---|---|---|
| 0 | 1.1 | 7 | 0.00% | aclnnIndex_IndexCheck_IndexCheck=13; aclnnIndex_IndexAiCore_Index=13; aclnnAdd_AddAiCore_Add=6 |
| 1 | 0.1 | 2 | 0.00% | _compute_slot_mapping_fused_groups_kernel=50; aclnnSubs_SubAiCore_Sub=2 |
| 2 | 28.3 | 336 | 0.09% | aclnnSparseFlashMlaMetadata_SparseFlashMlaMetadata_SparseFlashMlaMetadata=3003; _engram_host_uva_gather_dequant_kernel=2221; HcPre=1013 |
| 3 | 2.5 | 8 | 0.01% | hcom_allReduce__691_3_1=619; allreduceAicpuKernel=559; HcPost=557 |
| 4 | 15.6 | 58 | 0.05% | aclnnMatmul_MatMulV3Common_MatMulV3=5697; alltoallvAicpuKernel=1509; hcom_alltoallv__654_3_1=1133 |
| 5 | 1.2 | 3 | 0.00% | aclnnMatmul_MatMulCommon_MatMulV2=589; aclnnInplaceCopy_CastAiCore_Cast=379; aclnnInplaceCopy_SliceAiCore_Slice=277 |
| 6 | 2.3 | 10 | 0.01% | aclnnPowTensorScalar_PowsAiCore_Pows=1110; aclnnMean_ReduceMeanAiCore_ReduceMean=769; aclnnMul_MulAiCore_Mul=448 |
| 7 | 2.3 | 17 | 0.01% | aclnnMul_MulAiCore_Mul=1377; aclnnInplaceCopy_CastAiCore_Cast=408; aclnnReduceSum_ReduceSumOpAiCore_ReduceSum=349 |
| 8 | 18.6 | 113 | 0.06% | HcPre=2986; aclnnMatmul_MatMulCommon_MatMulV2=1322; alltoallvAicpuKernel=1292 |
| 9 | 0.0 | 1 | 0.00% | aclnnReduceSum_CastAiCore_Cast=9 |
| 10 | 644.5 | 2881 | 2.15% | SparseFlashMla=165818; HcPre=74368; alltoallvAicpuKernel=53509 |
| 11 | 3.2 | 4 | 0.01% | aclnnSoftmax_SoftmaxAiCore_SoftmaxV2=41; aclnnInplaceUniform_DSARandomUniform_DSARandomUniform=14; aclnnNeg_NegAiCore_Neg=7 |
| 12 | 2.1 | 8 | 0.01% | aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2=18; aclnnInplaceMaskedFillScalar_MaskedFillAiCore_MaskedFill=10; aclnnGeScalar_GreaterEqual_GreaterEqual=8 |
| 13 | 0.9 | 9 | 0.00% | _compute_slot_mapping_fused_groups_kernel=54; aclnnIndex_IndexAiCore_Index=17; aclnnIndex_IndexCheck_IndexCheck=13 |
| 14 | 232.1 | 1389 | 0.77% | SparseFlashMla=48409; HcPre=27795; alltoallvAicpuKernel=18233 |
| 15 | 464.4 | 2059 | 1.55% | SparseFlashMla=103767; HcPre=51537; VllmAscendQuantLightningIndexerV2=40366 |
| 16 | 78.7 | 649 | 0.26% | SparseFlashMla=8450; aclnnMatmul_MatMulV3Common_MatMulV3=8066; HcPre=8016 |
| 17 | 325.5 | 1545 | 1.09% | SparseFlashMla=79939; HcPre=39659; alltoallvAicpuKernel=28349 |
| 18 | 323.0 | 1254 | 1.08% | SparseFlashMla=63820; VllmAscendQuantLightningIndexerV2=59007; HcPre=31756 |
| 19 | 335.5 | 1813 | 1.12% | SparseFlashMla=68497; HcPre=37675; alltoallvAicpuKernel=24812 |
| 20 | 327.1 | 1308 | 1.09% | SparseFlashMla=67845; VllmAscendQuantLightningIndexerV2=66141; HcPre=34686 |
| 21 | 82.6 | 327 | 0.28% | VllmAscendQuantLightningIndexerV2=20403; SparseFlashMla=15959; HcPre=6918 |
| 22 | 771.2 | 3448 | 2.57% | SparseFlashMla=152392; VllmAscendQuantLightningIndexerV2=113064; HcPre=79380 |
| 23 | 362.7 | 1884 | 1.21% | SparseFlashMla=72496; HcPre=39661; alltoallvAicpuKernel=28395 |
| 24 | 314.0 | 1133 | 1.05% | VllmAscendQuantLightningIndexerV2=83368; SparseFlashMla=60005; HcPre=29756 |
| 25 | 111.7 | 431 | 0.37% | VllmAscendQuantLightningIndexerV2=25334; SparseFlashMla=19989; HcPre=9924 |
| 26 | 66.3 | 578 | 0.22% | aclnnMatmul_MatMulV3Common_MatMulV3=7460; HcPre=5962; SparseFlashMla=4451 |
| 27 | 712.6 | 2870 | 2.38% | SparseFlashMla=148146; VllmAscendQuantLightningIndexerV2=119240; HcPre=73408 |
| 28 | 789.5 | 3448 | 2.63% | SparseFlashMla=152642; VllmAscendQuantLightningIndexerV2=129374; HcPre=79326 |
| 29 | 800.8 | 3448 | 2.67% | SparseFlashMla=152787; VllmAscendQuantLightningIndexerV2=135786; HcPre=79333 |
| 30 | 791.1 | 3448 | 2.64% | SparseFlashMla=152696; VllmAscendQuantLightningIndexerV2=140428; HcPre=79474 |
| 31 | 813.3 | 3448 | 2.71% | SparseFlashMla=152851; VllmAscendQuantLightningIndexerV2=147430; HcPre=79376 |
| 32 | 770.8 | 3173 | 2.57% | VllmAscendQuantLightningIndexerV2=154051; SparseFlashMla=140840; HcPre=73397 |
| 33 | 0.3 | 3 | 0.00% | aclnnDynamicQuantV2_DynamicQuant_DynamicQuant=104; MoeTokenPermute=81; aclnnRepeatInterleave_RepeatInterleave_RepeatInterleave=22 |
| 34 | 877.4 | 3720 | 2.93% | SparseFlashMla=164974; VllmAscendQuantLightningIndexerV2=161042; HcPre=85360 |
| 35 | 930.4 | 4097 | 3.10% | VllmAscendQuantLightningIndexerV2=172918; SparseFlashMla=161416; HcPre=87270 |
| 36 | 744.8 | 2799 | 2.48% | VllmAscendQuantLightningIndexerV2=165166; SparseFlashMla=144568; HcPre=71347 |
| 37 | 837.1 | 3448 | 2.79% | VllmAscendQuantLightningIndexerV2=178818; SparseFlashMla=153009; HcPre=79208 |
| 38 | 843.0 | 3448 | 2.81% | VllmAscendQuantLightningIndexerV2=185906; SparseFlashMla=153194; HcPre=79345 |
| 39 | 876.8 | 3448 | 2.92% | VllmAscendQuantLightningIndexerV2=192222; SparseFlashMla=153185; HcPre=79393 |

## Top 25 kernel（按单次耗时）

- `VllmAscendQuantLightningIndexerV2`  49.52 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  49.52 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  49.51 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  49.51 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  48.41 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  48.40 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  48.40 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  48.40 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  47.67 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  47.66 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  47.66 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  47.64 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  46.48 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  46.47 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  46.46 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  46.45 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  45.52 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  45.52 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  45.52 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  45.51 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  44.46 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  44.45 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  44.45 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  44.45 ms  (bucket=indexer)
- `VllmAscendQuantLightningIndexerV2`  43.43 ms  (bucket=indexer)
