import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import glob
import re
import json
import time
import datetime
import asyncio
import logging
from tqdm import tqdm
from zhipuai import ZhipuAI
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建时间戳，用于文件名
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 创建结果文件夹
result_dir = f"evaluation_results_glm_{timestamp}"
os.makedirs(result_dir, exist_ok=True)

# 初始化智谱AI客户端
API_KEY = "dcee49f3b4a5477ebb1551dda9741fd8.U6KxpyfqnngElDyp"  # 替换为您的API密钥
client = ZhipuAI(api_key=API_KEY)

# 选择模型
MODEL = "glm-4-plus"

# 并发设置
MAX_CONCURRENT = 50  # 最大并发数
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)

# 速率限制设置
MAX_RETRIES = 5
RETRY_DELAY = 10  # 重试延迟（秒）

choices = ["A", "B", "C", "D"]

def build_prompt(text):
    return "[Round 1]\n\n问：{}\n\n答：".format(text)

extraction_prompt = '综上所述，ABCD中正确的选项是：'

# 辅助函数：处理API请求
def make_api_request(question, temperature=0.1, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": question}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                delay = RETRY_DELAY * (2 ** attempt)  # 指数退避
                logger.warning(f"API请求失败（尝试 {attempt+1}/{max_retries}）: {str(e)}. 等待 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                logger.error(f"所有重试都失败: {str(e)}")
                return None
    
    return None

# 提取选项的最终函数
def extract_option(response_text):
    if not response_text:
        return "A"  # 默认返回A
    
    # 多种模式匹配，尝试找出答案
    patterns = [
        r'正确的选项是[：:]?\s*([A-D])',  # 常见模式："正确的选项是A"
        r'选项\s*([A-D])\s*是正确的',     # "选项A是正确的"
        r'答案是\s*([A-D])',              # "答案是A"
        r'选择\s*([A-D])',                # "选择A"
        r'答案[为是:：]\s*([A-D])',       # "答案为A" 或 "答案是：A"
        r'应该[是选择:：]\s*([A-D])',     # "应该是A" 或 "应该选择A"
        r'\s([A-D])[.、是为]',            # " A." 或 "A是"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 如果上面都匹配不到，尝试直接找到独立的A、B、C、D
    match = re.search(r'\b([A-D])\b', response_text)
    if match:
        return match.group(1).upper()
    
    # 如果什么都找不到，返回默认值
    return "A"

# 异步处理单个样本
async def process_sample(item, dataset_name, result_file, total_samples, processed_count):
    text = item["inputs_pretokenized"]
    label = item["label"]
    correct_option = choices[label]
    
    try:
        # 提交API请求到线程池
        loop = asyncio.get_event_loop()
        first_response = await loop.run_in_executor(
            executor, 
            lambda: make_api_request(text)
        )
        
        # 构建包含提取提示的最终查询
        answer_text = text + "\n" + first_response + "\n" + extraction_prompt
        
        # 第二次API调用获取最终答案
        final_response = await loop.run_in_executor(
            executor, 
            lambda: make_api_request(answer_text) 
        )
        
        # 从回复中提取选项
        predicted_option = extract_option(final_response)
        predicted_index = choices.index(predicted_option) if predicted_option in choices else 0
        
        # 检查是否正确
        is_correct = (predicted_index == label)
        
        return {
            "text": text,
            "predicted": predicted_option,
            "actual": correct_option,
            "is_correct": is_correct,
            "processed": processed_count + 1,
            "response": first_response,
            "extraction": final_response
        }
        
    except Exception as e:
        logger.error(f"处理样本出错: {e}")
        return {
            "text": text,
            "predicted": "错误",
            "actual": correct_option,
            "is_correct": False,
            "processed": processed_count + 1,
            "error": str(e)
        }

# 加载保存的进度
def load_progress(result_file):
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            try:
                saved_data = json.load(f)
                return saved_data.get('correct', 0), saved_data.get('processed', 0)
            except:
                return 0, 0
    return 0, 0

# 保存进度
def save_progress(result_file, correct, processed, current_accuracy):
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'correct': correct,
            'processed': processed,
            'accuracy': current_accuracy,
            'last_processed_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

# 批量处理样本
async def process_batch(dataset, dataset_name, result_file):
    correct, processed = load_progress(result_file)
    logger.info(f"开始评估 {dataset_name}，已处理: {processed}/{len(dataset)}")
    
    # 跳过已处理的样本
    remaining_dataset = dataset[processed:]
    
    if not remaining_dataset:
        logger.info(f"{dataset_name} 已全部处理完成.")
        return correct, len(dataset)
    
    # 创建保存详细结果的文件
    details_file = f"{result_dir}/{dataset_name}_details.jsonl"
    
    # 准备任务列表
    tasks = []
    for i, item in enumerate(remaining_dataset):
        task = process_sample(item, dataset_name, result_file, len(dataset), processed + i)
        tasks.append(task)
    
    with tqdm(total=len(remaining_dataset), initial=0, desc=f"处理 {dataset_name}") as pbar:
        for i, future in enumerate(asyncio.as_completed(tasks)):
            result = await future
            
            # 更新计数
            if result.get("is_correct", False):
                correct += 1
            
            processed = result["processed"]
            current_accuracy = correct / processed if processed > 0 else 0
            
            # 保存进度
            save_progress(result_file, correct, processed, current_accuracy)
            
            # 保存这个样本的详细结果
            with open(details_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                "准确率": f"{current_accuracy:.4f}", 
                "正确": f"{correct}/{processed}"
            })
    
    final_accuracy = correct / len(dataset) if len(dataset) > 0 else 0
    logger.info(f"数据集 {dataset_name} 评估完成, 准确率: {final_accuracy:.4f}")
    
    return correct, len(dataset)

# 主要评估函数
async def run_evaluation():
    accuracy_dict, count_dict = {}, {}
    
    # 获取所有数据集文件
    dataset_files = sorted(glob.glob("./CEval/val/**/*.jsonl", recursive=True))
    
    for entry in dataset_files:
        dataset_name = os.path.basename(entry).replace(".jsonl", "")
        result_file = f"{result_dir}/{dataset_name}_results.json"
        
        # 加载数据集
        dataset = []
        with open(entry, encoding='utf-8') as file:
            for line in file:
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"无法解析行: {line}")
                    continue
        
        # 处理这个数据集
        correct, total = await process_batch(dataset, dataset_name, result_file)
        
        # 保存结果
        accuracy = correct / total if total > 0 else 0
        accuracy_dict[entry] = accuracy
        count_dict[entry] = total
    
    # 计算总体准确率
    acc_total, count_total = 0.0, 0
    for key in accuracy_dict:
        acc_total += accuracy_dict[key] * count_dict[key]
        count_total += count_dict[key]
    
    overall_accuracy = acc_total / count_total if count_total > 0 else 0
    
    # 保存总体评测结果
    summary_file = f"{result_dir}/summary_results.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("GLM-4-Plus 模型评测结果:\n")
        f.write("-" * 50 + "\n")
        for key in sorted(accuracy_dict.keys()):
            f.write(f"{key}: {accuracy_dict[key]:.4f} (样本数: {count_dict[key]})\n")
        
        f.write("\n" + "-" * 50 + "\n")
        f.write(f"总体准确率: {overall_accuracy:.4f}\n")
    
    logger.info(f"GLM-4-Plus 评测结果已保存到 {summary_file}")
    return overall_accuracy

# 测试连接性函数
def test_connection():
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "你好，世界！"}],
        )
        logger.info("模型连接测试成功!")
        return True
    except Exception as e:
        logger.error(f"模型连接测试失败: {e}")
        return False

# 主程序
if __name__ == "__main__":
    logger.info(f"开始 GLM-4-Plus 模型评测，最大并发数: {MAX_CONCURRENT}")
    
    if API_KEY == "your_api_key_here":
        logger.error("请设置有效的 API 密钥!")
        exit(1)
    
    if test_connection():
        # 运行异步主函数
        overall_accuracy = asyncio.run(run_evaluation())
        logger.info(f"评测完成！总体准确率: {overall_accuracy:.4f}")
    else:
        logger.error("无法连接到模型，请检查API密钥和网络连接后重试。")