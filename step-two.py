#测试训练好的模型

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "data"
# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建生成文本的管道
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, do_sample=True)

# 输入文本
ipt = "假期过得很愉快,放松了身心,我的感受如何？"

# 生成文本，设置截断，加载PeftModel之前
response_before = pipe(
    ipt,
    max_length=256,
    truncation=True,
    num_return_sequences=1,
    do_sample=True,  # 使用采样而不是贪婪解码
    temperature=0.7,  # 控制生成文本的随机性
    top_k=50,  # 保证生成文本的多样性
    top_p=0.95,  # 控制生成文本的质量
    repetition_penalty=2.0  # 减少重复生成的现象
)

print("Before loading PeftModel:", response_before)

# 加载PEFT模型
p_model = PeftModel.from_pretrained(model, model_id="output_model")

merge_model = p_model.merge_and_unload()

pipe = pipeline("text-generation", model=merge_model, tokenizer=tokenizer, do_sample=True)

# 输入文本
ipt = "假期过得很愉快,放松了身心,我的感受如何？"

# 生成文本，设置截断，加载PeftModel之后
response_after = pipe(
    ipt,
    max_length=256,
    truncation=True,
    num_return_sequences=1,
    do_sample=True,  # 使用采样而不是贪婪解码
    temperature=0.7,  # 控制生成文本的随机性
    top_k=50,  # 保证生成文本的多样性
    top_p=0.95,  # 控制生成文本的质量
    repetition_penalty=2.0  # 减少重复生成的现象
)

print("After loading PeftModel:", response_after)