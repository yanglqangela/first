from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from transformers.utils import hub
hub.TRANSFORMERS_CACHE = "./hf_cache"  # 可选，设置缓存目录

# 1. 加载RAG语料
with open("textprocess/RAG_data.txt", "r", encoding="utf-8") as f:
    rag_corpus = f.read().splitlines()

# 2. 加载嵌入模型
encoder_model = SentenceTransformer("all-MiniLM-L6-v2")  # 可替换为医学中文模型

# 3. 转换为语义向量
corpus_embeddings = encoder_model.encode(rag_corpus, convert_to_numpy=True)

# 4. 构建向量索引（FAISS）
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

# -- 2. 生成部分 --
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

local_model_path = "./hf_cache/Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
gen_model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True).float().cpu().eval()


def generate_sample(prompt):
    """生成样本并清理结果，确保只返回样本数据"""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids

    with torch.no_grad():
        outputs = gen_model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.95,
            temperature=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # 清理生成的结果
    sample = result.strip()
    
    # 移除可能的前缀
    prefixes = ["新样本：", "样本：", "生成结果：", "生成的样本：", "以下是", "这是", "sample：", "Sample："]
    for prefix in prefixes:
        if sample.lower().startswith(prefix.lower()):
            sample = sample[len(prefix):].strip()

    # 只取第一行有效内容
    sample = sample.split('\n')[0].strip()
    
    return sample

def retrieve_examples(query_text, top_k=3):
    """检索最相似的样本"""
    query_embedding = encoder_model.encode([query_text], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [rag_corpus[idx] for idx in indices[0]]

def extract_attributes(example):
    """从示例中提取关键属性"""
    attributes = {}
    
    # 提取年龄
    age_match = re.search(r'(\d+)岁', example)
    if age_match:
        attributes['age'] = age_match.group(1)
    
    # 提取性别
    if '男' in example:
        attributes['gender'] = '男性'
    elif '女' in example:
        attributes['gender'] = '女性'
    
    # 提取MoCA分数
    moca_match = re.search(r'MoCA[总分]*约?(\d+)分', example)
    if moca_match:
        attributes['moca'] = moca_match.group(1)    
    # 提取学历
    edu_keywords = ['文盲', '小学', '初中', '高中', '大专', '本科', '研究生']
    for edu in edu_keywords:
        if edu in example:
            attributes['education'] = edu
            break
    
    return attributes

def make_prompt_with_attributes(retrieved_examples, attributes):
    """创建提示，要求模型生成与原样本格式一致的新样本"""
    prompt = "请参考以下样本格式，生成一个新的认知功能下降老年人样本数据。"
    prompt += f"新样本应该是一位{attributes.get('age', '')}岁{attributes.get('gender', '')}"
    
    if 'moca' in attributes:
        prompt += f"，MoCA总分约{attributes['moca']}分"
    
    if 'education' in attributes:
        prompt += f"，学历为{attributes['education']}"
    
    prompt += "。\n\n请严格按照以下样本的格式生成，不要添加任何额外的标题、解释或分析：\n\n"
    
    for i, example in enumerate(retrieved_examples):
        prompt += f"{example}\n\n"
    
    prompt += "请直接生成一条新样本："
    return prompt

def generate_multiple_samples(num_samples=10):
    """生成多个样本并保存到文件"""
    new_samples = []
    attempts = 0
    max_attempts = num_samples * 3  # 增加尝试次数以确保生成足够数量的有效样本
    
    while len(new_samples) < num_samples and attempts < max_attempts:
        # 随机选择一个原始样本作为参考
        query_idx = attempts % len(rag_corpus)
        query = rag_corpus[query_idx]
        
        # 获取相似样本
        retrieved_examples = retrieve_examples(query, top_k=3)
        
        # 提取当前样本的属性
        attributes = extract_attributes(query)
        
        # 创建提示
        prompt = make_prompt_with_attributes(retrieved_examples, attributes)
        
        # 生成新样本
        generated_sample = generate_sample(prompt)
        
        attempts += 1
        
        # 验证样本质量，跳过无效样本
        if not (generated_sample and len(generated_sample) > 20 and "岁" in generated_sample 
                and ("男" in generated_sample or "女" in generated_sample) and "MoCA" in generated_sample):
            continue

        # 统一重命名样本
        sample_num = len(new_samples) + 1
        # 移除模型可能生成的 dataX， new_dataX， 等前缀
        cleaned_sample = re.sub(r'^(data|new_data)\d*，\s*', '', generated_sample)
        final_sample = f'样本{sample_num}，{cleaned_sample}'
        
        new_samples.append(final_sample)
        print(f"已生成 {len(new_samples)}/{num_samples} 个样本")

    # 保存生成的样本
    save_path = "./textprocess/RAG_new_data.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_samples))
    
    print(f"已生成 {len(new_samples)} 个新样本并保存至 {save_path}")

# 执行生成
generate_multiple_samples(num_samples=10)  # 可调整生成样本数量 