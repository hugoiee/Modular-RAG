"""
Query Construction Operators（查询构建）

论文核心技术：
- 将自然语言查询转换为结构化查询语言
- 支持多种数据源：SQL数据库、知识图谱等
- 实现跨模态的数据检索
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qwq import ChatQwen
from .base import BasePreRetrievalOperator


class TextToSQLOperator(BasePreRetrievalOperator):
    """
    Text-to-SQL 操作器

    功能：
    - 将自然语言问题转换为 SQL 查询
    - 支持结构化数据库的检索
    - 实现表格数据的精确查询

    应用场景：
    - 企业数据库查询
    - 数据分析和报表生成
    - 结构化知识库检索
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.0)
        self.schema = self.config.get("schema", None)  # 数据库schema信息

        # 初始化 LLM
        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
        )

    def execute(self, query: str) -> str:
        """
        将自然语言转换为 SQL

        Args:
            query: 自然语言查询

        Returns:
            SQL 查询语句
        """
        print(f"🗄️  Text-to-SQL: 正在生成SQL查询...")

        # 构建 prompt
        schema_info = self._format_schema() if self.schema else "请根据常见的数据库结构推断"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个SQL专家。将用户的自然语言问题转换为精确的SQL查询。

数据库Schema:
{schema_info}

要求：
1. 生成标准的SQL查询语句
2. 使用正确的表名和字段名
3. 添加必要的WHERE、JOIN等子句
4. 确保语法正确
5. 只输出SQL语句，不需要解释
6. 使用SELECT语句（不要使用修改数据的语句）

示例：
问题：查询所有销售额超过10000的订单
SQL：SELECT * FROM orders WHERE sales_amount > 10000;

问题：统计每个客户的订单总数
SQL：SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id;"""),
            ("human", "{query}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        sql_query = chain.invoke({
            "query": query,
            "schema_info": schema_info
        }).strip()

        # 清理可能的markdown代码块标记
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

        print(f"   自然语言: {query}")
        print(f"   SQL查询: {sql_query}")

        return sql_query

    def _format_schema(self) -> str:
        """格式化数据库schema信息"""
        if not self.schema:
            return "Schema信息未提供"

        # 简单的schema格式化
        if isinstance(self.schema, dict):
            formatted = []
            for table, columns in self.schema.items():
                formatted.append(f"表 {table}: {', '.join(columns)}")
            return "\n".join(formatted)
        else:
            return str(self.schema)


class TextToCypherOperator(BasePreRetrievalOperator):
    """
    Text-to-Cypher 操作器

    功能：
    - 将自然语言转换为 Cypher 查询（Neo4j图数据库）
    - 支持知识图谱的检索
    - 实现图结构数据的查询

    应用场景：
    - 知识图谱问答
    - 关系网络分析
    - 实体关系查询
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.0)
        self.graph_schema = self.config.get("graph_schema", None)

        # 初始化 LLM
        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
        )

    def execute(self, query: str) -> str:
        """
        将自然语言转换为 Cypher

        Args:
            query: 自然语言查询

        Returns:
            Cypher 查询语句
        """
        print(f"🕸️  Text-to-Cypher: 正在生成Cypher查询...")

        schema_info = self._format_graph_schema() if self.graph_schema else "请根据常见的图结构推断"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个Cypher查询专家。将用户的自然语言问题转换为Neo4j的Cypher查询。

知识图谱Schema:
{schema_info}

要求：
1. 生成标准的Cypher查询语句
2. 使用正确的节点标签和关系类型
3. 使用MATCH、WHERE、RETURN等子句
4. 确保语法正确
5. 只输出Cypher语句，不需要解释

示例：
问题：查找所有与"人工智能"相关的技术
Cypher：MATCH (t:Technology)-[:RELATED_TO]->(ai:Concept {{name: "人工智能"}}) RETURN t.name

问题：查找张三认识的所有人
Cypher：MATCH (p:Person {{name: "张三"}})-[:KNOWS]->(friend:Person) RETURN friend.name"""),
            ("human", "{query}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        cypher_query = chain.invoke({
            "query": query,
            "schema_info": schema_info
        }).strip()

        # 清理可能的markdown代码块标记
        cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()

        print(f"   自然语言: {query}")
        print(f"   Cypher查询: {cypher_query}")

        return cypher_query

    def _format_graph_schema(self) -> str:
        """格式化图schema信息"""
        if not self.graph_schema:
            return "Graph Schema信息未提供"

        if isinstance(self.graph_schema, dict):
            formatted = []
            if "nodes" in self.graph_schema:
                formatted.append(f"节点类型: {', '.join(self.graph_schema['nodes'])}")
            if "relationships" in self.graph_schema:
                formatted.append(f"关系类型: {', '.join(self.graph_schema['relationships'])}")
            return "\n".join(formatted)
        else:
            return str(self.graph_schema)


class MetadataFilterOperator(BasePreRetrievalOperator):
    """
    Metadata Filter 操作器

    功能：
    - 从查询中提取元数据过滤条件
    - 生成结构化的过滤器
    - 用于向量数据库的元数据过滤

    应用场景：
    - 基于时间、来源、类型等的过滤
    - 提高检索精度
    - 减少检索范围
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.0)
        self.available_metadata = self.config.get("available_metadata", [])

        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
        )

    def execute(self, query: str) -> Dict[str, Any]:
        """
        提取元数据过滤条件

        Args:
            query: 自然语言查询

        Returns:
            元数据过滤器字典
        """
        print(f"🔍 Metadata Filter: 正在提取过滤条件...")

        metadata_info = f"可用的元数据字段: {', '.join(self.available_metadata)}" if self.available_metadata else "请推断可能的元数据"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个元数据提取专家。从用户查询中提取结构化的元数据过滤条件。

{metadata_info}

要求：
1. 识别查询中的时间、来源、类型等限定条件
2. 输出JSON格式的过滤器
3. 使用标准的比较运算符（eq, ne, gt, lt, gte, lte, in）
4. 只输出JSON，不需要解释

示例：
查询：2024年关于人工智能的新闻
输出：{{"year": {{"eq": 2024}}, "topic": {{"eq": "人工智能"}}, "type": {{"eq": "新闻"}}}}

查询：来自路透社和彭博社的财经报道
输出：{{"source": {{"in": ["路透社", "彭博社"]}}, "category": {{"eq": "财经"}}}}"""),
            ("human", "{query}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        filter_json = chain.invoke({
            "query": query,
            "metadata_info": metadata_info
        }).strip()

        # 尝试解析JSON
        import json
        try:
            filter_dict = json.loads(filter_json)
            print(f"   提取的过滤条件: {filter_dict}")
            return filter_dict
        except json.JSONDecodeError:
            print(f"   ⚠️  无法解析过滤条件，返回空字典")
            return {}
