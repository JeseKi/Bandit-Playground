# Bandit Lib

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

一个用于模拟和评估多臂老虎机（Multi-Armed Bandit）算法的 Python 库。

## 简介

Bandit Lib 提供了一套灵活的工具，用于实现、测试和比较不同的多臂老虎机策略。无论您是在学术研究、在线学习系统还是推荐系统中探索 Bandit 算法，本库都能帮助您快速搭建实验并可视化结果。

## 特性

- **多种算法实现**: 内置了多种经典 Bandit 算法：
  - **Greedy**: 贪婪算法。
  - **UCB (Upper Confidence Bound)**: 上置信界算法。
  - **Thompson Sampling**: 汤普森采样。
- **灵活的环境配置**:
  - 支持**静态环境**和**动态环境**，在动态环境中，臂的奖励概率会随时间变化。
  - 可自定义臂的数量和奖励机制。
- **高效的批量训练**:
  - 使用 `multiprocessing` 并行运行多次模拟，以获得稳健的统计结果。
- **丰富的性能指标**:
  - 自动计算和记录关键指标，如：`regret`（悔憾）、`reward`（奖励）、最优臂选择率等。
- **强大的可视化**:
  - 使用 Plotly 生成交互式图表，方便地可视化单个实验的指标或比较多个算法的性能。

## 项目结构

```
bandit/
├── docs/                   # 文档
├── bandit_lib/             # 核心库代码
│   ├── agents/             # Bandit 代理和算法实现
│   ├── env/                # 模拟环境
│   ├── runner/             # 用于运行实验的高级工具
│   └── utils/              # 工具函数，包括日志记录和可视化
├── examples/               # Jupyter Notebook 示例
└── README.md
```

## 安装
<div style="background:#f0f8ff; border-left:5px solid #4682b4; padding:10px; margin:10px 0;">
  <strong>💡 Note</strong>
  <p>Python: 3.10.18</p>
  <p>操作系统：Ubuntu 22.04 LTS 桌面版</p>
</div>

您可以通过 poetry 来安装依赖：

```bash
pip install poetry==2.2.1
poetry install
```

## 快速开始

您可以参考这个[文档](docs/quick_start.md)

## 示例

`examples/` 目录下包含了更详细的 Jupyter Notebook 示例，展示了如何：
- 运行不同的算法并比较它们的性能。
- 在动态环境中评估算法。
- 自定义和保存可视化结果。

## 许可证

本项目使用 [MIT License](LICENSE) 授权。
