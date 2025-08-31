import json

from transformers import AutoTokenizer,AutoModelForCausalLM


##lora微调 文件 问答


model_name = "data"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name).to('mps')

print("---加载完成---")

# from data_prepare import samples
# with open("datasets.jsonl","w",encoding="utf-8") as f:
#     for s in samples:
#         json_line = json.dumps(s, ensure_ascii=False)
#         f.write(json_line+"\n")
#     else:
#         print("---数据集完成处理---")


from datasets import load_dataset
dataset = load_dataset("json",data_files={"train": "datasets.jsonl"}, split="train")
print("数据数量:",len(dataset))
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
print(f"train dataset len: {len(train_dataset)}")
print(f"test dataset len: {len(eval_dataset)}")


def tokenizer_function(many_samples):
    texts =[f"{question}\n{answer}" for question, answer in
        zip(many_samples["question"], many_samples["answer"])]
    tokens = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    # print(tokens)
    tokens["labels"]= tokens["input_ids"].copy()
    return tokens

tokenized_train_dataset = train_dataset.map(tokenizer_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenizer_function, batched=True)
print("---完成tokenizing--")

# 第五步：量化设置
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
#cpu加载模型
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
#GPU加载模型
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
print("---已经完成量化模型的加载--")

# 第六步:lora微调设置
from peft import get_peft_model, LoraConfig, TaskType
lora_config = LoraConfig(
    r=1, lora_alpha=16, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("---lora微调设置完毕---")

from transformers import TrainingArguments,Trainer
training_args = TrainingArguments(
    output_dir="./finetuned_models",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    fp16=False,
    logging_steps=100,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    learning_rate=3e-5,
    logging_dir="./logs",
    run_name="deepseek-r1-distill-finetune"
)

trainer = Trainer(
    model = model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)

trainer.train()

model.save_pretrained("./output_model")

'''执行结果
/Users/liuhongya/2025/llama/deepseek-fine-tuning/.venv/bin/python /Users/liuhongya/2025/llama/deepseek-fine-tuning/step-one.py 
/Users/liuhongya/2025/llama/deepseek-fine-tuning/.venv/lib/python3.8/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
---加载完成---
数据数量: 50
train dataset len: 45
test dataset len: 5
Map: 100%|██████████| 45/45 [00:00<00:00, 2370.88 examples/s]
---完成tokenizing--
Map: 100%|██████████| 5/5 [00:00<00:00, 1379.34 examples/s]
---已经完成量化模型的加载--
/Users/liuhongya/2025/llama/deepseek-fine-tuning/.venv/lib/python3.8/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
trainable params: 136,192 || all params: 1,777,224,192 || trainable%: 0.0077
---lora微调设置完毕---
100%|██████████| 50/50 [8:05:30<00:00, 582.62s/it]
{'train_runtime': 29130.8404, 'train_samples_per_second': 0.015, 'train_steps_per_second': 0.002, 'train_loss': 15.4565869140625, 'epoch': 9.48}

Process finished with exit code 0
'''


''''
D:\workspace\PycharmProjects\deepseek-fine-tuning\.venv\Scripts\python.exe D:\workspace\PycharmProjects\deepseek-fine-tuning\step-one.py 
---加载完成---
Generating train split: 50 examples [00:00, 8543.07 examples/s]
数据数量: 50
train dataset len: 45
test dataset len: 5
Map: 100%|██████████| 45/45 [00:00<00:00, 3053.02 examples/s]
Map: 100%|██████████| 5/5 [00:00<00:00, 839.47 examples/s]
---完成tokenizing--
---已经完成量化模型的加载--
The 8-bit optimizer is not available on your device, only available on CUDA for now.
trainable params: 136,192 || all params: 1,777,224,192 || trainable%: 0.0077
---lora微调设置完毕---
  0%|          | 0/60 [00:00<?, ?it/s]D:\workspace\PycharmProjects\deepseek-fine-tuning\.venv\Lib\site-packages\ torch\ utils\data\dataloader.py:666: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
100%|██████████| 60/60 [35:03<00:00, 35.06s/it]
{'train_runtime': 2103.3283, 'train_samples_per_second': 0.214, 'train_steps_per_second': 0.029, 'train_loss': 16.105989583333333, 'epoch': 10.0}

Process finished with exit code 0
'''''


'''
D:\workspace\PycharmProjects\deepseek-fine-tuning\.venv\Scripts\python.exe D:\workspace\PycharmProjects\deepseek-fine-tuning\step-one.py 
---加载完成---
数据数量: 50
train dataset len: 45
test dataset len: 5
Map: 100%|██████████| 45/45 [00:00<00:00, 3508.25 examples/s]
Map: 100%|██████████| 5/5 [00:00<00:00, 1013.12 examples/s]
---完成tokenizing--
---已经完成量化模型的加载--
trainable params: 136,192 || all params: 1,777,224,192 || trainable%: 0.0077
---lora微调设置完毕---
100%|██████████| 60/60 [02:27<00:00,  2.45s/it]
{'train_runtime': 147.0914, 'train_samples_per_second': 3.059, 'train_steps_per_second': 0.408, 'train_loss': 16.10968017578125, 'epoch': 10.0}

Process finished with exit code 0
'''