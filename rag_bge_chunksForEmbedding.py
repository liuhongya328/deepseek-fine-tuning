from rag_bge_loadPdfForChunks import load_pdf_for_chunks
from sentence_transformers import SentenceTransformer
import os

def load_embedding_model():
    """
    加载bge-small-zh-v1.5模型
    :return: 返回加载的bge-small-zh-v1.5模型
    """
    # SentenceTransformer读取绝对路径下的bge-small-zh-v1.5模型，非下载
    embedding_model = SentenceTransformer(os.path.abspath('./models/bge-small-zh-v1.5'))
    return embedding_model

def chunks_to_embedding(chunks ,embedding_model):
# 文本块转化为嵌入向量列表，normalize_embeddings表示对嵌入向量进行归一化，用于准确计算相似度
    embeddings = []
    for chunk in chunks:
        embedding = embedding_model.encode(chunk, normalize_embeddings=True)
        embeddings.append(embedding)

    return embeddings

print("加载PDF文档并分割文本块")
pdf_file = "files/AI赋能：AI重新定义产品经理.pdf"
chunks = load_pdf_for_chunks(pdf_file)
print("加载Embedding模型")
embedding_model = load_embedding_model()
print("文本块转化为嵌入向量列表")
embeddings = chunks_to_embedding(chunks, embedding_model)
print(f"嵌入向量列表长度: {len(embeddings)}")

#https://zhuanlan.zhihu.com/p/4785788322