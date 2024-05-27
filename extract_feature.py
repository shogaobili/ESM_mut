import torch
import pandas as pd
import numpy as np
from transformers import EsmModel, EsmTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from accelerate import Accelerator
import gc

# 初始化Accelerator
accelerator = Accelerator()

# 定义模型和标记器的本地路径
ESM_path = "/data/huggingfaceesm/esm2_t36_3B_UR50D/"
data_train_path = "cdna_train.csv"

df = pd.read_csv(data_train_path,nrows=1000)
df = df[["mut_seq","ddg"]]
print(df)

# 定义提取特征函数
def extract_features(model_path,data_df, batch_size=8):
    # 确保CUDA可用，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
    # model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D")
    model = EsmModel.from_pretrained(model_path)
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model.to(device) # 将模型迁移到GPU或CPU
    model = accelerator.prepare(model) # 准备模型以使用accelerator

    total_feature = []
    for i in range(0, len(data_df)):
        mut_seq = data_df["mut_seq"][i]
        ddg = data_df["ddg"][i]
        # Tokenization with specified padding and truncation
        inputs = tokenizer(mut_seq, return_tensors="pt", padding=True, truncation=True, max_length=512)
        #key: attention_mask val.shape: torch.Size([8, 61])
        # transfer data to GPU or CPU
        inputs = {key: val.to(accelerator.device) for key, val in inputs.items()}
        print(f"seq {i} inputs moved to device: {accelerator.device}")
        with torch.no_grad():
            # use autocast to automatically choose the best data type for the model
            with accelerator.autocast():
                outputs = model(**inputs)
                
                # get the last hidden state mean value
        seq_feature = outputs.last_hidden_state.mean(dim=1).cpu().numpy() # transfer data back to CPU
        seq_feat_lable = np.insert(seq_feature, 0, ddg).reshape((2561,1))
        total_feature.append(seq_feat_lable)
        print(f"seq {i} features extracted")
        # 清除缓存并释放显存
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
    # 将所有特征序列连接起来
    total_feature = np.concatenate(total_feature, axis=0).reshape(len(data_df),2561)
    np.save('./ESM_feat_lable.npy', total_feature)
    print(total_feature)
    return total_feature
# 提取特征
features_list = extract_features(ESM_path, df)
