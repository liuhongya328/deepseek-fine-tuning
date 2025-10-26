# 从 langchain_community.document_loaders 模块中导入各种类型文档加载器类
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredHTMLLoader,
)
# 引入操作系统库，后续配置环境变量与获得当前文件路径使用
import os

def load_document(file_path):
    """
    解析各种文档格式的文件，返回文档内容字符串
    :param file_path: 文档文件路径
    :return: 返回文档内容的字符串
    """

    # 定义文档解析加载器字典，根据文档类型选择对应的文档解析加载器类和输入参数
    DOCUMENT_LOADER_MAPPING = {
        ".pdf": (PDFPlumberLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf8"}),
        ".doc": (UnstructuredWordDocumentLoader, {}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        ".ppt": (UnstructuredPowerPointLoader, {}),
        ".pptx": (UnstructuredPowerPointLoader, {}),
        ".xlsx": (UnstructuredExcelLoader, {}),
        ".csv": (CSVLoader, {}),
        ".md": (UnstructuredMarkdownLoader, {}),
        ".xml": (UnstructuredXMLLoader, {}),
        ".html": (UnstructuredHTMLLoader, {}),
    }

    ext = os.path.splitext(file_path)[1]  # 获取文件扩展名，确定文档类型
    loader_tuple = DOCUMENT_LOADER_MAPPING.get(ext)  # 获取文档对应的文档解析加载器类和参数元组

    if loader_tuple: # 判断文档格式是否在加载器支持范围
        loader_class, loader_args = loader_tuple  # 解包元组，获取文档解析加载器类和参数
        loader = loader_class(file_path, **loader_args)  # 创建文档解析加载器实例，并传入文档文件路径
        documents = loader.load()  # 加载文档
        content = "\n".join([doc.page_content for doc in documents])  # 多页文档内容组合为字符串
        print(f"文档 {file_path} 的部分内容为: {content[:100]}...")  # 仅用来展示文档内容的前100个字符
        # print(f"文档 {file_path} 的部分内容为: {content}") # 输出全部文档内容
        return content  # 返回文档内容的多页拼合字符串

    print(file_path+f"，不支持的文档类型: '{ext}'") # 若文件格式不支持，输出信息，返回空字符串。
    return ""

# 单个文件
#file = './数字化转型.pdf'
#load_document(file)

# 遍历目录
def load_files(directory):
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                print(entry.path)
                load_document(entry.path)
                print("====================================================")
            #elif entry.is_dir(): 子目录下递归变量
            #    list_files(entry.path)

# 示例用法
directory = './files'
load_files(directory)