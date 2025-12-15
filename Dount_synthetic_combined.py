import os
import random
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from transformers import DonutProcessor, VisionEncoderDecoderModel

# -----------------------------
# 1️ 설정
# -----------------------------
OUTPUT_IMG_DIR = "output/images"
OUTPUT_LABEL_DIR = "output/labels"
FONT_DIR = "fonts"
CANVAS_W, CANVAS_H = 900, 3000
FONT_SIZE_MIN, FONT_SIZE_MAX = 28, 48
MIN_COLUMNS, MAX_COLUMNS = 6, 12
BATCH_SIZE = 4
EPOCHS = 5
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REAL_IMG_DIR = "real/images"
REAL_LABEL_DIR = "real/labels"

# -----------------------------
# 2️ 폰트 로드
# -----------------------------
def load_fonts(font_dir=FONT_DIR):
    font_files = [os.path.join(font_dir,f) for f in os.listdir(font_dir) if f.lower().endswith((".ttf",".otf"))]
    if not font_files:
        raise RuntimeError(f"No fonts found in {font_dir}")
    return font_files

FONT_FILES = load_fonts()
def pick_font(size):
    return ImageFont.truetype(random.choice(FONT_FILES), size)

# -----------------------------
# 3️ 족보 텍스트 랜덤 생성
# -----------------------------
FAMILY_NAMES = ["張","李","王","趙","錢","孫","周","吳","鄭","劉","陳","楊","黃"]
GIVEN_NAME_SINGLE = list("中華志文國仁義忠孝昌良成景天明")
GIVEN_NAME_DOUBLE = ["世傑","天佑","永昌","文俊","光遠","志偉","敬業","紹宗"]
OFFICES = ["朝議郎","太史令","司徒","太常","承事郎","都尉","典史","書吏","州牧","主簿","典客","史官","里正","鄉紳","教授","祭祀"]
HEAVENLY_STEMS = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸"]
EARTHLY_BRANCHES = ["子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"]

def gen_person_entry():
    family = random.choice(FAMILY_NAMES)
    given = random.choice(GIVEN_NAME_SINGLE) if random.random() < 0.6 else random.choice(GIVEN_NAME_DOUBLE)
    name = family + given
    ganzi = random.choice(HEAVENLY_STEMS) + random.choice(EARTHLY_BRANCHES)
    born = random.randint(1700,1920)
    died = born + random.randint(20,80)
    if died > 2020:
        died = born + random.randint(20,70)
    office = random.choice(OFFICES) if random.random()<0.5 else ""
    return {"name":name,"ganzi":ganzi,"office":office,"born":str(born),"died":str(died)}

# -----------------------------
# 4️ 세로 컬럼 렌더링
# -----------------------------
def render_vertical_column(draw, start_x, col_top, col_width, entry_texts, font, char_spacing=6):
    words=[]
    y=col_top
    char_x = start_x + col_width//2
    for txt in entry_texts:
        for ch in txt:
            w,h = font.getsize(ch)
            x1 = int(char_x - w/2)
            y1 = int(y)
            x2 = x1 + w
            y2 = y1 + h
            draw.text((x1,y1), ch, font=font, fill=(0,0,0))
            words.append({"text":ch, "bbox":[x1,y1,x2,y2]})
            y += h + char_spacing
        y += int(char_spacing*1.5)
    return words, y

def merge_chars_to_words(char_words, y_gap_thresh=20):
    if not char_words:
        return []
    for w in char_words:
        x1,y1,x2,y2 = w["bbox"]
        w["cx"] = (x1+x2)/2
        w["cy"] = (y1+y2)/2
    char_words_sorted = sorted(char_words, key=lambda z:(-z["cx"],z["cy"]))
    groups=[]
    used=[False]*len(char_words_sorted)
    for i,cw in enumerate(char_words_sorted):
        if used[i]:
            continue
        group=[cw]
        used[i]=True
        for j in range(i+1,len(char_words_sorted)):
            if used[j] or abs(char_words_sorted[j]["cx"]-cw["cx"])>20:
                continue
            if char_words_sorted[j]["cy"]-group[-1]["cy"]<(cw["bbox"][3]-cw["bbox"][1])+y_gap_thresh:
                group.append(char_words_sorted[j])
                used[j]=True
        texts="".join([g["text"] for g in group])
        x1 = min([g["bbox"][0] for g in group])
        y1 = min([g["bbox"][1] for g in group])
        x2 = max([g["bbox"][2] for g in group])
        y2 = max([g["bbox"][3] for g in group])
        groups.append({"text":texts,"bbox":[int(x1),int(y1),int(x2),int(y2)]})
    return groups

# -----------------------------
# 5️ augmentation
# -----------------------------
def add_old_paper_texture(img_np):
    noise = np.random.normal(0,12,img_np.shape).astype(np.int16)
    img_np = np.clip(img_np.astype(np.int16)+noise,0,255).astype(np.uint8)
    tint = np.array([255,245,220],dtype=np.uint8)
    img_np = (img_np*0.9 + tint*0.1).astype(np.uint8)
    return img_np

def add_vertical_scratches(img_np, count=5):
    h,w = img_np.shape[:2]
    for _ in range(count):
        x=random.randint(0,w-1)
        thickness=random.randint(1,3)
        color=random.randint(150,255)
        cv2.line(img_np,(x,0),(x,h),(color,color,color),thickness)
    return img_np

def vertical_downsample_blur(img_np):
    h,w = img_np.shape[:2]
    down=cv2.resize(img_np,(w,h//2))
    up=cv2.resize(down,(w,h))
    return up

# -----------------------------
# 6️ synthetic 페이지 생성
# -----------------------------
def generate_synthetic_page():
    img_np = np.ones((CANVAS_H,CANVAS_W,3),dtype=np.uint8)*255
    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)

    n_cols = random.randint(MIN_COLUMNS,MAX_COLUMNS)
    col_widths = [int(CANVAS_W//n_cols*random.uniform(0.85,1.05)) for _ in range(n_cols)]
    xs=[]
    cur_x = CANVAS_W - col_widths[0] - 40
    xs.append(cur_x)
    for i in range(1,n_cols):
        cur_x = cur_x - col_widths[i] - random.randint(8,18)
        xs.append(cur_x)

    all_words=[]
    donut_persons=[]
    for col_idx in range(n_cols):
        x = xs[col_idx]
        w = col_widths[col_idx]
        font_size=random.randint(FONT_SIZE_MIN,FONT_SIZE_MAX)
        font=pick_font(font_size)

        n_entries=random.randint(6,15)
        entry_texts=[]
        column_persons=[]
        for _ in range(n_entries):
            p = gen_person_entry()
            lines=[p["name"],p["ganzi"]]
            if p["office"]:
                lines.append(p["office"])
            lines.append(p["born"]+" - "+p["died"])
            entry_texts.extend(lines)
            column_persons.append(p)
        top_margin=random.randint(50,150)
        words,_ = render_vertical_column(draw,start_x=x,col_top=top_margin,col_width=w,entry_texts=entry_texts,font=font,char_spacing=int(font_size*0.12))
        all_words.extend(words)
        donut_persons.extend(column_persons)

    img_np = np.array(img_pil)
    img_np = add_old_paper_texture(img_np)
    if random.random()<0.5:
        img_np = add_vertical_scratches(img_np, random.randint(3,10))
    if random.random()<0.4:
        img_np = vertical_downsample_blur(img_np)

    words_grouped = merge_chars_to_words(all_words)
    donut_target={"persons":donut_persons}

    return img_np, words_grouped, donut_target

# -----------------------------
# 7️ Mixed Dataset (Synthetic + Real)
# -----------------------------
class MixedGenealogyDataset(Dataset):
    def __init__(self, n_synth_samples, real_img_dir=None, real_label_dir=None, transform=None):
        self.n_synth_samples = n_synth_samples
        self.real_img_dir = real_img_dir
        self.real_label_dir = real_label_dir
        self.transform = transform

        self.real_files = []
        if real_img_dir and os.path.exists(real_img_dir):
            self.real_files = [f for f in os.listdir(real_img_dir) if f.endswith((".jpg",".png"))]
            self.real_files.sort()

        self.total_len = n_synth_samples + len(self.real_files)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.n_synth_samples:
            return self.generate_synthetic_item()
        else:
            return self.load_real_item(idx - self.n_synth_samples)

    def generate_synthetic_item(self):
        img_np, words, donut_target = generate_synthetic_page()
        img = Image.fromarray(img_np)
        labels = json.dumps(donut_target, ensure_ascii=False)
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "labels": labels}

    def load_real_item(self, real_idx):
        if not self.real_files:
            raise IndexError("No real images found")
        img_name = self.real_files[real_idx]
        img_path = os.path.join(self.real_img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        label_path = os.path.join(self.real_label_dir, img_name.rsplit(".",1)[0]+".json")
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                label = json.load(f)
        else:
            label = {"persons":[]}
        labels = json.dumps(label, ensure_ascii=False)
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "labels": labels}

# -----------------------------
# 8️ DataLoader
# -----------------------------
transform = T.Compose([
    T.Resize((512,512)),
    T.ToTensor(),
])

dataset = MixedGenealogyDataset(
    n_synth_samples=50,
    real_img_dir=REAL_IMG_DIR,
    real_label_dir=REAL_LABEL_DIR,
    transform=transform
)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# 9️ Donut Fine-tune
# -----------------------------
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base").to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = processor.tokenizer(batch["labels"], return_tensors="pt", padding=True).input_ids.to(DEVICE)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")
