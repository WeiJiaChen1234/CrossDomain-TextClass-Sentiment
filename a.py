import torch
model_path = r"saved_dict/bert_source.ckpt"
try:
    state_dict = torch.load(model_path)
    print("文件加载成功，包含参数：", list(state_dict.keys())[:5])  # 打印前5个参数名
except Exception as e:
    print("加载失败：", e)