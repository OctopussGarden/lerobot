import torch
from safetensors.torch import load_file, save_file
import os

# ================= 配置路径 =================
# 你原来的权重文件路径 (根据你的日志修改)
input_model_path = "models/pi05_base/model.safetensors"
# 输出的新权重文件路径
output_model_path = "models/pi05_base_fixed/model.safetensors"
# ===========================================

print(f"正在加载权重: {input_model_path} ...")
state_dict = load_file(input_model_path)

new_state_dict = {}
renamed_count = 0

print("正在处理 Key 映射...")

for key, value in state_dict.items():
    new_key = key
    
    # 修复规则：将 .dense.weight 替换为 .weight (针对 LayerNorm)
    # 你的报错里主要是 input_layernorm 和 post_attention_layernorm
    
    if "input_layernorm.dense.weight" in key:
        new_key = key.replace("input_layernorm.dense.weight", "input_layernorm.weight")
    elif "input_layernorm.dense.bias" in key:
        new_key = key.replace("input_layernorm.dense.bias", "input_layernorm.bias")
        
    elif "post_attention_layernorm.dense.weight" in key:
        new_key = key.replace("post_attention_layernorm.dense.weight", "post_attention_layernorm.weight")
    elif "post_attention_layernorm.dense.bias" in key:
        new_key = key.replace("post_attention_layernorm.dense.bias", "post_attention_layernorm.bias")
        
    elif "norm.dense.weight" in key:
        new_key = key.replace("norm.dense.weight", "norm.weight")
    elif "norm.dense.bias" in key:
        new_key = key.replace("norm.dense.bias", "norm.bias")

    # 如果发生替换，打印日志
    if new_key != key:
        print(f"修改: {key} -> {new_key}")
        renamed_count += 1
    
    new_state_dict[new_key] = value

print(f"处理完成！共修改了 {renamed_count} 个 Key。")

# 确保输出目录存在
os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

print(f"正在保存新权重到: {output_model_path} ...")
save_file(new_state_dict, output_model_path)
print("✅ 成功！请修改你的训练脚本指向这个新路径。")
