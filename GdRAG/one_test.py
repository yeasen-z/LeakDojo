from transformers import AutoTokenizer, AutoModelForCausalLM

# 选择模型
model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/Qwen/Qwen2.5-14B-Instruct-1M"

# 加载 tokenizer 和模型
print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)
print("模型加载完成！")

# --- 这里是您要测试的简单例子 ---
text_content = "input: I woke up this morning feeling the whole room is spinning when i was sitting down. I went to the bathroom walking unsteadily, as i tried to focus i feel nauseous. I try to vomit but it wont come out.. After taking panadol and sleep for few hours, i still feel the same.. By the way, if i lay down or sit down, my head do not spin, only when i want to move around then i feel the whole world is spinning.. And it is normal stomach discomfort at the same time? Earlier after i relieved myself, the spinning lessen so i am not sure whether its connected or coincidences.. Thank you doc!\noutput: Hi, Thank you for posting your query. The most likely cause for your symptoms is benign paroxysmal positional vertigo (BPPV), a type of peripheral vertigo. In this condition, the most common symptom is dizziness or giddiness, which is made worse with movements. Accompanying nausea and vomiting are common. The condition is due to problem in the ear, and improves in a few days on own. Betahistine tablets would help relieve your symptoms. Doing vestibular rehabilitation or adaptation exercises would prevent the recurrence of these symptoms. An ENT evaluation would also help. I hope it helps. Best wishes, Chat Doctor."
user_question = f"""
你是一个医学知识图谱抽取助手。
请从以下文本中抽取 **症状 (Symptom)**、**疾病 (Disease)**、**药物 (Drug)**，并建立它们的关系。
只返回 JSON，确保符合以下格式：

{{
  "nodes": [
    {{"id": "symptom_1", "type": "Symptom", "name": "muscle cramp"}},
    {{"id": "disease_1", "type": "Disease", "name": "heart attack"}},
    {{"id": "drug_1", "type": "Drug", "name": "Panadol"}}
  ],
  "relationships": [
    {{"source": "symptom_1", "target": "disease_1", "type": "possible_sign_of"}},
    {{"source": "drug_1", "target": "symptom_1", "type": "relieves"}}
  ]
}}

要求：
- 如果文本中出现症状（如头晕、恶心、肌肉痉挛），归类为 **Symptom**
- 如果出现疾病（如心脏病、胃炎、流感），归类为 **Disease**
- 如果出现药物（如Panadol、阿司匹林），归类为 **Drug**
- 关系类型尽量使用以下集合：
  - "possible_sign_of"（症状 → 疾病）
  - "relieves"（药物 → 症状）
  - "treats"（药物 → 疾病）

文本: {text_content}
"""

# 构造对话消息
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_question}
]

# 使用 tokenizer 格式化输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 生成回复
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
)

# 解码并打印回复
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"\n👤 用户: {user_question}")
print(f"🤖 Qwen: {response}")