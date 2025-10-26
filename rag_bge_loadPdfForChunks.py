from langchain_community.document_loaders import PyPDFLoader # PDF文档提取
from langchain_text_splitters import RecursiveCharacterTextSplitter # 文档拆分chunk

# 定义方法，加载PDF文档并分割文本块
def load_pdf_for_chunks(pdf_file):
    # PyPDFLoader加载PDF文件，忽略图片提取
    pdf_loader = PyPDFLoader(pdf_file, extract_images=False)
    # 配置RecursiveCharacterTextSplitter分割文本块库参数，每个文本块的大小为512字符（非token），相邻文本块之间的重叠128字符（非token）
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    # 加载PDF文档,提取所有页的文本内容
    pdf_content_list = pdf_loader.load()
    # 将每页的文本内容用换行符连接，合并为PDF文档的完整文本
    pdf_text = "\n".join([page.page_content for page in pdf_content_list])
    print(f"PDF文档的总字符数: {len(pdf_text)}")

    # 将PDF文档文本分割成文本块Chunk
    chunks = text_splitter.split_text(pdf_text)
    print(f"分割的文本Chunk数量: {len(chunks)}")

    for chunk in chunks:
        print(f"Chunk的字符数: {len(chunk)}")
        print(chunk)
        print("\n")

    return chunks

# 测试方法
# pdf_file = "files/AI赋能：AI重新定义产品经理.pdf"
# load_pdf_for_chunks(pdf_file)