# file: donut_chinese_pretrain_finetune.py
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset
from PIL import Image

# ---------------------------
# 1) 설정
# ---------------------------
MODEL_NAME = "naver-clova-ix/donut-base"   # 출발점(공식 base). HuggingFace에서 불러옴. :contentReference[oaicite:3]{index=3}
OUTPUT_DIR = "./donut-chinese-pretrained"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_IMG_DIR = "output/images"
OUTPUT_LABEL_DIR = "output/labels"

# ---------------------------
# 2) Processor / Model 로드
# ---------------------------
processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)

# (선택) 중국어 특화 토큰/스페셜 토큰 추가
# 예: 모델이 출력할 JSON 프리/포스트 앵커 토큰을 정의했다면 추가.
special_tokens = {"additional_special_tokens": ["<s_answer>", "<s_json>", "</s_json>"]}
if hasattr(processor, "tokenizer"):
    processor.tokenizer.add_special_tokens(special_tokens)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

# ---------------------------
# 3) Dataset 클래스 (HuggingFace dataset 혹은 jsonl 사용 가능)
# 각각의 item: {"image": "<path or PIL>", "label": "<string json/text>"}
# ---------------------------
class GenealogyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        label_path = os.path.join(self.label_dir, self.files[idx].replace(".jpg",".json"))
        with open(label_path, "r", encoding="utf-8") as f:
            label = json.load(f)
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": label}

    def __len__(self):
        return len(self.files)

# HuggingFace Dataset wrapper
class DonutChineseDataset(Dataset):
    def __init__(self, hf_dataset, processor, max_target_length=512, image_column="image", label_column="label"):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_target_length = max_target_length
        self.image_column = image_column
        self.label_column = label_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # 이미지 로드 (path이면 open), hf dataset에서 이미 image PIL로 로드되는 경우도 있음
        image = item[self.image_column]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # label: 이미 문자열(JSON문자열 또는 플레인 텍스트)
        label = item[self.label_column]

        # processor는 pixel_values + labels(tokenized) 반환 (DonutProcessor는 image->pixel_values, tokenizer->labels 함수 제공)
        # 아래는 transformer 버전에 따라 인자명이 다를 수 있으므로 필요시 적절히 변경.
        encoding = self.processor(image, text_target=label, return_tensors="pt")

        # HuggingFace Trainer expects plain tensors, remove batch dim
        pixel_values = encoding["pixel_values"].squeeze()
        labels = encoding["labels"].squeeze()

        return {"pixel_values": pixel_values, "labels": labels}

# ---------------------------
# 4) 데이터 불러오기 예시
# - continued pretrain: 큰 unlabeled_text dataset (이미 라벨로 전체 문장 텍스트 있음)
# - finetune: label이 JSON string 으로 되어 있는 dataset
# ---------------------------
# 예: local jsonl 포맷 (each line: {"image":"/path/1.jpg","label":"{\"invoice_date\":\"...\"}"})
def load_jsonl_dataset(path):
    # HuggingFace datasets로 불러오는 것이 제일 편함
    return load_dataset("json", data_files=path, split="train")

# 사용자 경로
pretrain_jsonl = "./data/pretrain_chinese.jsonl"
finetune_jsonl = "./data/finetune_chinese.jsonl"

pretrain_ds = load_jsonl_dataset(pretrain_jsonl)
finetune_ds = load_jsonl_dataset(finetune_jsonl)

# HuggingFace Dataset -> PyTorch Dataset wrapper
train_pretrain = DonutChineseDataset(pretrain_ds, processor, image_column="image", label_column="label")
train_finetune = DonutChineseDataset(finetune_ds, processor, image_column="image", label_column="label")

# ---------------------------
# 5) TrainingArguments (continued pretraining)
# ---------------------------
pretrain_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "continued_pretrain"),
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=100,
    save_total_limit=3,
    save_steps=1000,
    remove_unused_columns=False,
    predict_with_generate=False,   # pretraining phase는 generate 없이 teacher forcing으로 학습
)

# ---------------------------
# 6) Trainer (continued pretraining)
# ---------------------------
# Note: labels are already token ids (processor produces 'labels'); ensure data collator returns dicts
from transformers import default_data_collator

def collate_fn(batch):
    # batch is list of {"pixel_values":tensor, "labels":tensor}
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}

pretrain_trainer = Seq2SeqTrainer(
    model=model,
    args=pretrain_args,
    train_dataset=train_pretrain,
    data_collator=collate_fn,
)

# ---------------------------
# 7) Run continued pretraining
# ---------------------------
pretrain_trainer.train()
pretrain_trainer.save_model(os.path.join(OUTPUT_DIR, "donut-chinese-pretrained"))

# ---------------------------
# 8) Fine-tuning 설정 (supervised)
# ---------------------------
# 로드한 pretrained checkpoint를 기반으로 미세조정
model = VisionEncoderDecoderModel.from_pretrained(os.path.join(OUTPUT_DIR, "donut-chinese-pretrained"))
model.to(DEVICE)

finetune_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "finetune"),
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=3e-5,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    predict_with_generate=True,   # 평가시 generate 한다면 True
    evaluation_strategy="steps",
    eval_steps=500,
    remove_unused_columns=False,
)

finetune_trainer = Seq2SeqTrainer(
    model=model,
    args=finetune_args,
    train_dataset=train_finetune,
    data_collator=collate_fn,
)

finetune_trainer.train()
finetune_trainer.save_model(os.path.join(OUTPUT_DIR, "donut-chinese-finetuned"))
