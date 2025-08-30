##对输入的进行向量化

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

model_name = "data"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('mps')
config = AutoConfig.from_pretrained(model_name,trust_remote_code=True,device_map ='auto')
sentence="中国的首都是北京"
input_ids = tokenizer(sentence,return_tensors="pt",max_length=128,truncation=True,padding=True,
                      add_special_tokens=True)
print(input_ids)
target_ids = tokenizer(sentence,return_tensors="pt",max_length=128,truncation=True,padding=True,
                      add_special_tokens=False)['input_ids'][0].tolist()+[config.eos_token_id]
print(target_ids)


input_ids = tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True, padding=True,
add_special_tokens=True)
target_ids = tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True, padding=True,
add_special_tokens=False)['input_ids'][0].tolist()+[config.eos_token_id]
#输入长度
input_length=len(input_ids['input_ids'][0].tolist())
#目标长度
target_length=len(target_ids)
start=input_length-target_length
output=model.cal_logits(**input_ids,return_dict=True,output_attentions=False,
output_hidden_states=False)[0]
results=[]
# for i in range(0,target_length):
