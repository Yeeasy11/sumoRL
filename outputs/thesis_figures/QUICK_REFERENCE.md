# 快速参考卡 (Quick Reference Card)

## 8大图表总结

| # | 文件名 | 关键数据 | 论文位置 |
|----|-------|--------|---------|
| **E1** | `E1_baseline_validation.png` | 零碰撞验证 ✅ | Sec 4.1导言 |
| **E3** | `E3_method_comparison.png` | 旅行时间↓68.9% | Sec 4.1主体 |
| **E4** | `E4_flow_robustness.png` | 流量900vh/h下优势78% | Sec 4.2鲁棒性 |
| **E5** | `E5_ablation_study.png` | 邻车信息174x重要 | Sec 4.3消融 |
| **H1** | `congestion_heatmap.png` | 规则拥堵集中vs PPO均匀 | Sec 4.1补充 |
| **H2** | `waiting_time_analysis.png` | CDF+统计表+分布对比 | Sec 4.1补充 |
| **T1** | `trajectory_spacetime.png` | 锯齿形vs平滑流 | Sec 4.4安全 |
| **T2** | `ttc_analysis.png` | 碰撞风险↓13.8%, 方差↓82.9% | Sec 4.4安全 |

---

## 一句话总结各图

- **E1**: "我们的方法没有增加风险" (安全基准)
- **E3**: "我们的改进很大" (主要对比) ⭐ **最重要**  
- **E4**: "流量越大我们越好" (鲁棒性)
- **E5**: "邻车信息很关键" (模型正当性)
- **H1-H2**: "交通流更均衡了" (机制洞察)
- **T1-T2**: "行驶安全性提升了" (安全论证) ⭐ **投稿会看**

---

## 按论文Section分配

```
Abstract
├─ E3主要指标

1. Introduction  
├─ (无图)

2. Related Work
├─ (无图)

3. Method
├─ (无图，文字说明)

4. Experiments ⭐
├─ 4.1 Performance
│  ├─ 图: E1(安全验证) + E3(主对比) + H1-H2(热力)
│  └─ 文: "PPO改进68.9%..."
│
├─ 4.2 Robustness  
│  ├─ 图: E4(多流量)
│  └─ 文: "流量900时优势78%..."
│
├─ 4.3 Ablation
│  ├─ 图: E5(消融)
│  └─ 文: "邻车信息删除致174x恶化..."
│
└─ 4.4 Safety Analysis
   ├─ 图: T1(轨迹) + T2(TTC)
   └─ 文: "TTC中位数2.06s, 碰撞风险↓13.8%..."

5. Conclusion
├─ (无图)
└─ 引用E1-E5结果来支撑结论
```

---

## "老板要看什么？" 答案

| 问题 | 答案 | 图表 |
|------|------|------|
| 有没有改进？ | **旅行时间减68.9%** | **E3** |
| 是不是唬人？ | 流量900v/h下优势78%，E4证实 | **E4** |
| 会不会有风险？ | 零碰撞，且TTC方差↓82.9% | **E1 + T2** |
| 为啥能成功？ | 邻车信息174x重要 | **E5** |
| 对比是否公平？ | 相同条件、相同种子、详见方法章 | E1开头 |

---

## 数据验证清单

| 指标 | Rule | PPO | ✓ Pass |
|------|------|-----|--------|
| 采样量 | 15,608 | 14,602 | ✓ 均衡 |
| 碰撞 | 0 | 0 | ✓ 安全 |
| 种子 | 1 | 1 | ✓ 一致 |
| 时长 | 600s | 600s | ✓ 等长 |
| 配置 | 相同 | 相同 | ✓ 可控 |

---

## 制作脚本速查

若要重新绘制某个图：

```bash
# E1-E5主图  
cd f:\py\sumo-rl
conda run -n sumo-rl python experiments\generate_thesis_figures.py

# 热力图
conda run -n sumo-rl python experiments\generate_heatmap.py

# 轨迹-TTC
conda run -n sumo-rl python experiments\plot_trajectory_ttc.py \
  --route sumo_rl/nets/single-intersection/single-intersection.rou.xml \
  --net sumo_rl/nets/single-intersection/single-intersection.net.xml \
  --seconds 600 --delta-time 1 --seed 1 \
  --ppo-model models/ppo_final.zip \
  --outdir outputs/thesis_figures
```

---

## 论文写作模板 (Copy-Paste Ready)

### 4.1 性能对比段落
```
基于单交叉口分布式PPO强化学习算法的实验结果表明(见图3)，
与规则基础控制相比，本方法在多个关键指标上取得显著改进：
平均旅行时间从1168.3秒降低至365.2秒(↓68.9%)，
等待时间从843.2秒大幅下降至42.1秒(↓95.0%)，
系统吞吐量从578.2 veh/h提升至595.6 veh/h(↑3.0%)，
平均速度从4.31 m/s增加至5.10 m/s(↑18.1%)。
同时，安全性评估表明两种方法均未发生碰撞事件(图1)，
证明了提议的强化学习方法在保证安全的前提下实现了交通效率的显著提升。
```

### 4.4 安全性段落  
```
为进一步评估安全性，本文采用时间裕度(TTC)指标对两种方法进行深度比较。
基于600秒模拟中的在线TraCI采样，PPO相比规则基础方法：
(1) 碰撞风险(TTC<1.0s)的样本占比从32.60%降至28.11%，下降13.8%；
(2) TTC标准差从831.5秒大幅降至141.9秒，下降82.9%，
    表明PPO的车间间距分布显著更加集中稳定；
(3) 从时空轨迹看，规则基础方法表现出明显的周期性拥堵特征(锯齿形波动)，
    而PPO则保持均匀的流量分布，缓解了上游排队。
综合表明，PPO通过学习自适应信号调整，有效维持了更稳定的车间间距，
从而兼具了效率改进与安全保障。
```

---

## 投稿前检查清单

- [ ] 8个PNG文件都在 `outputs/thesis_figures/` 中
- [ ] 所有PNG分辨率 ≥ 300 DPI(投稿要求)
- [ ] 各图均无水印/草稿标记
- [ ] 标题/坐标轴/图例字体≥10pt(易读)
- [ ] 数据表格格式一致(小数点位数统一)
- [ ] 论文段落中引用了所有8个图
- [ ] 没有遗漏的重要数据或对比
- [ ] 与Abstract中的数据一致(68.9%等)

---

**生成**: 2025年3月23日  
**用途**: 论文写作速查、老板演讲、会议展示
