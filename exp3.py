from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda"
model_id = "openai-community/gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

from datasets import load_dataset

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

# 打印test的基本信息和前几行内容
print("Test dataset info:")
print(test.info)
print("\nFirst few lines of test dataset:")
for i in range(5):
    print(test["text"][i])  
# 打印encodings的基本信息和前几行内容
print("\nEncodings info:")
print(encodings)
print("\nFirst few lines of encodings:")
for i in range(5):
    print(encodings["input_ids"][i])

# 打印encodings的基本信息和前几行内容
print("Encodings info:")


import torch
from tqdm import tqdm

max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)
print(seq_len)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())


list=nlls
print("list的基本信息：")
print("元素个数:", len(list))
print("元素类型:", type(list[0]))

# 打印list的前几个元素
print("\nlist的前几个元素：")
for i in range(3):
    print(list[i])