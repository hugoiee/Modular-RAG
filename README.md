# Modular-RAG

模块化 RAG：将 RAG 系统转变为乐高式可重构框架，论文实现。[论文链接](https://arxiv.org/html/2407.21059v1)

```shell
uv sync
```

实现思路与框架：
1. 使用 LangChain 完成六大模块(索引、检索前、检索、检索后、生成、编排)
2. 使用 LangGraph 实现 RAG 编排模式

模型选择，Qwen 系列模型
