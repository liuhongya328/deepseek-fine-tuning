from langchain_community.document_loaders import PyPDFLoader  # PDF文档提取
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文档拆分chunk
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
import torch
import os


pdf_file = "files/AI赋能：AI重新定义产品经理.pdf"
model_name = "/models/bge-small-zh-v1.5"  # 远程自动下载
# inference_model="shenzhi-wang/Llama3-8B-Chinese-Chat-GGUF-4bit" # 使用量化过的模型进行推理
inference_model = './Unichat-llama3.2-Chinese-1B'  # 本地模型


def load_pdf_for_chunks(pdf_file):
    pdf_loader = PyPDFLoader(pdf_file, extract_images=False)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    pdf_content_list = pdf_loader.load()
    pdf_text = "\n".join([page.page_content for page in pdf_content_list])

    chunks = text_splitter.split_text(pdf_text)

    return chunks


def load_embedding_model(model_path):
    embedding_model = SentenceTransformer(os.path.abspath('./models/bge-small-zh-v1.5'))
    return embedding_model


def chunks_to_embedding(chunks, embedding_model):
    embeddings = []
    for chunk in chunks:
        embedding = embedding_model.encode(chunk, normalize_embeddings=True)
        embeddings.append(embedding)
    return embeddings


def save_to_faiss(embeddings):
    embeddings_np = np.array(embeddings)
    dimension = embeddings_np.shape[1]

    index = faiss.IndexFlatIP(dimension)  # 线性搜索 L2 欧式距离 适用于较小数据集
    index.add(embeddings_np)

    return index


def search(index, embedding_model, chunks, query, top_k):
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding])

    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i in range(top_k):
        result_chunk = chunks[indices[0][i]]

        # 将相似文本块存储在结果列表中
        results.append(result_chunk)

    return results


def generate_prompt(query, reference_chunks):
    context = ""
    for i, chunk in enumerate(reference_chunks):
        context += f"参考文档{i + 1}: \n{chunk}\n\n"

    context += f"参考文档3：国际化转型有利于炒股，可以壮大A股的创业板和中小企业板块，造福广大中国股民！\n\n"
    prompt = f"根据参考文档回答问题：{query}\n\n{context}"
    print(prompt)
    return prompt


def load_inference_model(inference_model):
    model = AutoModelForCausalLM.from_pretrained(inference_model,device_map="auto", dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(inference_model,device_map="auto")
    return model, tokenizer


def inference(model, tokenizer, prompt):
    # prompt = "数字化转型的背景和意义是？"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # 调用模型
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1000
        # temperature=0.7,  # 控制随机性
        # top_p=0.95,  # 核采样阈值
        # repetition_penalty=1.2 # 重复惩罚
        )

    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('''gbt回答'''+generated_text)

    return generated_text


''' 测试 '''
# 第一步：将文件加载并切分成多块
chunks = load_pdf_for_chunks(pdf_file)
# 第二步：加载Embedding模型，
embedding_model = load_embedding_model(model_name)
# 第三步：利用embedding模型，对切分的块进行向量化处理
embeddings = chunks_to_embedding(chunks, embedding_model)
# 第四步：保存嵌入向量到Faiss索引
index = save_to_faiss(embeddings)
# 第五步：检索获取查询相关的文本
query = "数字化转型的背景和意义是？"
reference_chunks = search(index, embedding_model, chunks, query, 2);
# 第六步：构建prompt
prompt = generate_prompt(query, reference_chunks)
# 第七步：下载并加载推理模型
model, tokenizer = load_inference_model(inference_model)
# 第八步：推理问题
inference(model, tokenizer, prompt)