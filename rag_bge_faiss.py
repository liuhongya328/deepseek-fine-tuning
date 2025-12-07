import faiss
import numpy as np
from rag_bge_chunksForEmbedding import chunks_to_embedding, load_embedding_model
from rag_bge_loadPdfForChunks import load_pdf_for_chunks


def save_to_faiss(embeddings):
    # 将嵌入向量列表转化为numpy数组，Faiss向量库操作需要numpy数组输入
    embeddings_np = np.array(embeddings)

    # 获取嵌入向量的维度（每个向量的长度），shape返回数组的形状，第二个元素是列数（即每个嵌入向量的维度）
    dimension = embeddings_np.shape[1]

    # 使用余弦相似度创建FAISS索引，创建了一个基于内积的索引对象 (IndexFlatIP)，用于执行相似度搜索
    # IndexFlatIP 表示这是一个“平坦”的内积索引，它会存储所有向量并在查询时遍历整个数据库来计算内积，从而找到最相似的向量。这种方式简单直接，但在非常大的数据集上可能不是最高效的。
    index = faiss.IndexFlatIP(dimension) # 线性搜索 L2 欧式距离 适用于较小数据集
    # 将所有的嵌入向量添加到FAISS索引中，后续可以用来进行相似性检索
    index.add(embeddings_np)

    return index

def search(index, embedding_model,chunks,query, top_k):
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    # 将嵌入向量转化为numpy数组，Faiss索引操作需要numpy数组输入
    query_embedding = np.array([query_embedding])
    # 设置返回最相似的前k个文本块，indices表示最相似文本块的索引，需要结合chunks列表获取文本块的原始内容
    distances, indices = index.search(query_embedding, top_k)
    results=[]
    print(f"最相似的前{top_k}个文本块:")
    for i in range(top_k):
        # 获取相似文本块的原始内容
        result_chunk = chunks[indices[0][i]]
        print(f"文本块 {i}:\n{result_chunk}")

        # 获取相似文本块的相似度得分
        result_distance = distances[0][i]
        print(f"相似度得分: {result_distance}\n")

        # 将相似文本块存储在结果列表中
        results.append(result_chunk)

    print("检索过程完成.")
    return results

print("加载PDF文档并分割文本块")
pdf_file = "files/AI赋能：AI重新定义产品经理.pdf"
chunks = load_pdf_for_chunks(pdf_file)
print("加载Embedding模型")
embedding_model = load_embedding_model()
print("文本块转化为嵌入向量列表")
embeddings = chunks_to_embedding(chunks, embedding_model)
print("保存嵌入向量到Faiss索引")
index = save_to_faiss(embeddings)

# 查询文本
query = "数字化转型的背景和意义是？"
result=search(index, embedding_model, chunks, query, 3);
# 输出查询结果
print(result)