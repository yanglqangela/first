from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. 加载RAG语料
with open("textprocess/RAG_data.txt", "r", encoding="utf-8") as f:
    rag_corpus = f.read().splitlines()

# 2. 加载嵌入模型
model = SentenceTransformer("all-MiniLM-L6-v2")  # 可替换为医学中文模型

# 3. 转换为语义向量
corpus_embeddings = model.encode(rag_corpus, convert_to_numpy=True)

# 4. 构建向量索引（FAISS）
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

# 用户输入查询条件，例如想生成一个类似“MoCA为20左右的女性样本”
query_text = "82岁女性，MoCA总分约20分，独居，学历为初中"

# 1. 转换为语义向量
query_embedding = model.encode([query_text], convert_to_numpy=True)

# 2. 执行语义检索（返回前3条最相似记录）
top_k = 3
distances, indices = index.search(query_embedding, top_k)

# 3. 输出检索结果
for i, idx in enumerate(indices[0]):
    print(f"Top {i+1}:\n{rag_corpus[idx]}\n")

# 拼接 Prompt 示例
retrieved_examples = [rag_corpus[idx] for idx in indices[0]]

prompt = "以下是三位认知功能下降的老年样本资料，请你基于它们的特征生成一个新但合理的结构化社区老人记录：\n\n"
for i, example in enumerate(retrieved_examples):
    prompt += f"【样本{i+1}】{example}\n"
prompt += "\n请生成一个新的样本："

print(prompt)
