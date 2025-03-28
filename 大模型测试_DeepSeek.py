import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import glob
import re
import json
import torch
import torch.utils.data
from tqdm import tqdm
from openai import OpenAI
import datetime
import time

# 创建时间戳，用于文件名
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 创建结果文件夹
result_dir = f"evaluation_results_deepseek_{timestamp}"
os.makedirs(result_dir, exist_ok=True)

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-9cddedf701b541c8bf8bdc678fc21180",  # DeepSeek API 密钥
    base_url="https://api.deepseek.com/v1",
)

# 选择模型
model = "deepseek-chat"

# API限制相关变量
RPM_LIMIT = 3  # 每分钟请求数限制
SECONDS_PER_REQUEST = 60.0 / RPM_LIMIT  # 每个请求至少间隔20秒
last_request_time = time.time() - SECONDS_PER_REQUEST  # 上次请求时间

choices = ["A", "B", "C", "D"]

def build_prompt(text):
    return "[Round 1]\n\n问：{}\n\n答：".format(text)

extraction_prompt = '综上所述，ABCD中正确的选项是：'

# 辅助函数：处理API请求并遵循速率限制
def make_api_request(messages, temperature=0.3, max_retries=3):
    global last_request_time
    
    # 确保请求间隔符合RPM限制
    time_since_last = time.time() - last_request_time
    if time_since_last < SECONDS_PER_REQUEST:
        wait_time = SECONDS_PER_REQUEST - time_since_last
        print(f"等待 {wait_time:.2f} 秒以遵守API限制...")
        time.sleep(wait_time)
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            last_request_time = time.time()  # 更新最后请求时间
            return completion.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # 指数退避
                print(f"遇到速率限制，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise e
    
    return None  # 所有重试都失败

# 辅助函数：从回复中提取选项
def extract_option_from_response(response_text):
    if not response_text:
        return "A"  # 默认返回A
        
    # 检查回复中是否包含"ABCD中正确的选项是"后面的字符
    match = re.search(r'正确的选项是[：:]?\s*([A-D])', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # 尝试寻找"选项X是正确的"这样的模式
    match = re.search(r'选项\s*([A-D])\s*是正确的', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # 尝试直接找到单独的A、B、C、D选项
    match = re.search(r'\b([A-D])\b', response_text)
    if match:
        return match.group(1).upper()
    
    # 如果无法提取，则默认返回A
    return "A"

# 主评估逻辑
accuracy_dict, count_dict = {}, {}
for entry in glob.glob("./CEval/val/**/*.jsonl", recursive=True):
    dataset_name = os.path.basename(entry).replace(".jsonl", "")
    result_file = f"{result_dir}/{dataset_name}_results.txt"
    
    # 尝试从之前的结果加载
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            correct = saved_data.get('correct', 0)
            processed = saved_data.get('processed', 0)
    else:
        correct = 0
        processed = 0
    
    # 加载数据集
    dataset = []
    with open(entry, encoding='utf-8') as file:
        for line in file:
            dataset.append(json.loads(line))
    
    # 跳过已处理的样本
    remaining_dataset = dataset[processed:]
    
    print(f"开始评估 {entry}，已处理: {processed}/{len(dataset)}")
    
    for i, item in enumerate(tqdm(remaining_dataset, initial=processed, total=len(dataset))):
        text = item["inputs_pretokenized"]
        label = item["label"]
        
        try:
            # 第一次API调用获取回复
            first_response = make_api_request([
                {"role": "user", "content": text}
            ])
            
            # 构建包含提取提示的最终查询
            answer_text = text + "\n" + first_response + "\n" + extraction_prompt
            
            # 第二次API调用获取最终答案
            final_response = make_api_request([
                {"role": "user", "content": answer_text}
            ])
            
            # 从回复中提取选项
            predicted_option = extract_option_from_response(final_response)
            predicted_index = choices.index(predicted_option) if predicted_option in choices else 0
            
            # 检查是否正确
            if predicted_index == label:
                correct += 1
                
            # 更新进度并保存
            processed += 1
            with open(result_file, 'w', encoding='utf-8') as f:
                current_accuracy = correct / processed if processed > 0 else 0
                json.dump({
                    'correct': correct,
                    'processed': processed,
                    'accuracy': current_accuracy,
                    'last_processed_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
                
            # 打印当前状态
            print(f"样本 {processed}/{len(dataset)}, 预测: {predicted_option}, 实际: {choices[label]}, 正确: {predicted_index == label}, 当前准确率: {correct/processed:.4f}")
                
        except Exception as e:
            print(f"处理样本 {processed+1} 时出错: {e}")
            # 记录错误但继续下一个样本
            continue
    
    # 计算并保存此数据集的准确率
    accuracy = correct / len(dataset) if len(dataset) > 0 else 0
    accuracy_dict[entry] = accuracy
    count_dict[entry] = len(dataset)
    print(f"数据集 {entry} 评估完成, 准确率: {accuracy:.4f}")

# 计算总体准确率
acc_total, count_total = 0.0, 0
for key in accuracy_dict:
    acc_total += accuracy_dict[key] * count_dict[key]
    count_total += count_dict[key]

overall_accuracy = acc_total / count_total if count_total > 0 else 0

# 保存总体评测结果
summary_file = f"{result_dir}/summary_results.txt"
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("DeepSeek模型评测结果:\n")
    f.write("-" * 50 + "\n")
    for key in accuracy_dict:
        f.write(f"{key}: {accuracy_dict[key]:.4f} (样本数: {count_dict[key]})\n")
    
    f.write("\n" + "-" * 50 + "\n")
    f.write(f"总体准确率: {overall_accuracy:.4f}\n")

print(f"DeepSeek评测结果已保存到 {summary_file}")