import openai
import json
from pathlib import Path
import time
from tqdm import tqdm  # 导入tqdm
import os  # 导入os模块

version = "20240121-Jul"

# 设置 OpenAI API 配置
openai.api_base = "http://localhost:8000/v1"  # 本地API端点
openai.api_key = "sk-xxx"  # 替换为你的API密钥

def generate_batch(examples, model: str):
    """使用OpenAI API生成批处理结果"""
    
    for ex in tqdm(examples, desc="生成中", unit="条"):
        # 构造消息格式
        messages = [
            {"role": "system", "content": "You are an expert programming assistant."},
            {"role": "user", "content": ex['prompt_sft']}
        ]
        
        # API调用
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=8192,
                top_p=0.95
            )
            ex['output'] = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成时发生错误: {str(e)}")
            ex['output'] = ""
            time.sleep(5)  # 错误后稍作等待
        
    return examples

def generate_main(data_path: str, model_name: str, saved_path: str, cot: bool = False):
    # 创建输出目录（如果不存在的话）
    output_dir = os.path.dirname(saved_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取数据
    examples = [json.loads(x) for x in open(data_path).readlines()]
    
    # 处理COT逻辑
    if cot:
        def _convert_for_sft(ex):
            ex['prompt_sft'] = ex["prompt_sft"] + "\nPlease first outline the steps and then write the solution."
            return ex
        examples = [_convert_for_sft(x) for x in examples]
        saved_path = saved_path.replace(".jsonl", ".cot.jsonl")
    
    # 执行生成
    generated_examples = generate_batch(examples, model_name)
    
    # 保存结果
    with open(saved_path, 'w', encoding='utf-8') as fw:
        for ex in generated_examples:
            fw.write(json.dumps(ex) + '\n')
    print(f"已保存{len(generated_examples)}条结果至{saved_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, 
                       default=Path(__file__).parent.joinpath(f"data/{version}.jsonl").as_posix())
    parser.add_argument('--model_name', type=str, 
                       default='DeepSeek-R1-Distill-Qwen-32B')
    parser.add_argument('--saved_path', type=str, 
                       default=f'output/{version}.openai-output.jsonl')
    parser.add_argument('--cot', action='store_true', default=False)
    args = parser.parse_args()

    generate_main(
        data_path=args.data_path,
        model_name=args.model_name,
        saved_path=args.saved_path,
        cot=args.cot
    )
