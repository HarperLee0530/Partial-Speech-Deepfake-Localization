# 结果目录

`historical_evidence` 为旧实验的必要统计与复核，不能与缩减后结果混为一谈。`migration` 为数据与程序验收。

- [sal_baseline_20261112](sal_baseline_20261112)：缩减 SAL 训练完成后的冻结模型、dev/train_edit/train_review 指标、区间与错误诊断。
- [sal_frozen_test_20261113](sal_frozen_test_20261113)：同一冻结模型的 PS eval（3,562 条）与 LPS（3,800 条）测试结果、逐录音预测缓存和核验记录。EER 分别为 4.503134% 与 37.909355%；伪造类帧级 F1 分别为 95.383669% 与 62.441524%。

新方法收益网络尚未实现，也没有据这些基线结果证明 LPS 泛化改善。
