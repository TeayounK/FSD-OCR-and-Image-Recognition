import os
import random
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import imgaug.augmenters as iaa
import imgaug as ia

# -----------------------------
# 설정
# -----------------------------
INPUT_IMG_DIR = "data/testImages"   # 실제 스캔 이미지 폴더
OUT_IMG_DIR = "output/images"
OUT_LABEL_DIR = "output/labels"
FONT_DIR = "fonts"              # CJK 폰트 넣어두는 폴더
AUG_PER_IMAGE = 3               # 이미지당 synthetic 생성 개수

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# Canvas 크기 (실제 이미지에 맞게 resize 가능)
CANVAS_W = 900
CANVAS_H = 3000

# 폰트 사이즈
FONT_SIZE_MIN = 28
FONT_SIZE_MAX = 48

# 컬럼 설정
MIN_COLUMNS = 6
MAX_COLUMNS = 12

# 족보 데이터 샘플
FAMILY_NAMES = ["張","李","王","趙","錢","孫","周","吳","鄭","劉","陳","楊","黃"]
GIVEN_NAME_SINGLE = list("中華志文國仁義忠孝昌良成景天明")
GIVEN_NAME_DOUBLE = ["世傑","天佑","永昌","文俊","光遠","志偉","敬業","紹宗"]
OFFICES = ["朝議郎","太史令","司徒","太常","承事郎","都尉","典史","書吏","州牧","主簿","典客","史官","里正","鄉紳","教授","祭祀"]
HEAVENLY_STEMS = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸"]
EARTHLY_BRANCHES = ["子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"]

random.seed(42)

# -----------------------------
# 폰트 로드
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
# 족보 인물 랜덤 생성
# -----------------------------
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
# 세로 텍스트 컬럼 렌더링
# -----------------------------
def render_vertical_column(draw, start_x, col_top, col_width, entry_texts, font, char_spacing=6):
    words = []
    y = col_top
    char_x = start_x + col_width // 2
    for txt in entry_texts:
        for ch in txt:
            w,h = font.getsize(ch)
            x1 = int(char_x - w/2)
            y1 = int(y)
            x2 = int(x1 + w)
            y2 = int(y1 + h)
            draw.text((x1,y1),ch,font=font,fill=(0,0,0))
            words.append({"text":ch,"bbox":[x1,y1,x2,y2]})
            y += h + char_spacing
        y += int(char_spacing*1.5)
    return words, y

# -----------------------------
# 문자 단위 -> 단어 단위 병합
# -----------------------------
def merge_chars_to_words(char_words, y_gap_thresh=20):
    if not char_words:
        return []
    for w in char_words:
        x1,y1,x2,y2 = w["bbox"]
        w["cx"] = (x1+x2)/2
        w["cy"] = (y1+y2)/2
    char_words_sorted = sorted(char_words,key=lambda z:(-z["cx"],z["cy"]))
    groups = []
    used = [False]*len(char_words_sorted)
    for i,cw in enumerate(char_words_sorted):
        if used[i]:
            continue
        group = [cw]
        used[i]=True
        for j in range(i+1,len(char_words_sorted)):
            if used[j]:
                continue
            if abs(char_words_sorted[j]["cx"]-cw["cx"])<20:
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
# 종이 노이즈/스크래치
# -----------------------------
def add_old_paper_texture(img_np):
    noise = np.random.normal(0,12,img_np.shape).astype(np.int16)
    img_np = np.clip(img_np.astype(np.int16)+noise,0,255).astype(np.uint8)
    tint = np.array([255,245,220],dtype=np.uint8)
    img_np = (img_np*0.9+tint*0.1).astype(np.uint8)
    return img_np

def add_vertical_scratches(img_np, count=5):
    h,w = img_np.shape[:2]
    for _ in range(count):
        x = random.randint(0,w-1)
        thickness=random.randint(1,3)
        color=random.randint(150,255)
        cv2.line(img_np,(x,0),(x,h),(color,color,color),thickness)
    return img_np

def vertical_downsample_blur(img_np):
    h,w = img_np.shape[:2]
    down = cv2.resize(img_np,(w,h//2))
    up = cv2.resize(down,(w,h))
    return up

# -----------------------------
# synthetic 생성 (실제 이미지 + 족보 텍스트)
# -----------------------------
def generate_synthetic_page(base_img):
    H,W = base_img.shape[:2]
    img = base_img.copy()
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # 컬럼 수, 위치
    n_cols = random.randint(MIN_COLUMNS,MAX_COLUMNS)
    col_widths = [int(W//n_cols*random.uniform(0.85,1.05)) for _ in range(n_cols)]
    xs=[]
    cur_x = W - col_widths[0] - 40
    xs.append(cur_x)
    for i in range(1,n_cols):
        cur_x = cur_x - col_widths[i] - random.randint(8,18)
        xs.append(cur_x)

    all_words=[]
    donut_persons=[]
    for col_idx in range(n_cols):
        x = xs[col_idx]
        w = col_widths[col_idx]
        fsize=random.randint(FONT_SIZE_MIN,FONT_SIZE_MAX)
        font=pick_font(fsize)

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
        words,_=render_vertical_column(draw,start_x=x,col_top=top_margin,col_width=w,entry_texts=entry_texts,font=font,char_spacing=int(fsize*0.12))
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
# 전체 폴더 처리
# -----------------------------
def create_synthetic_dataset_from_real(input_img_dir, out_img_dir, out_label_dir, aug_per_image=3):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    for fname in os.listdir(input_img_dir):
        if not fname.lower().endswith((".png",".jpg",".jpeg")):
            continue
        img_path = os.path.join(input_img_dir,fname)
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        for i in range(aug_per_image):
            synth_img, words, donut = generate_synthetic_page(img)
            base=fname.rsplit(".",1)[0]
            out_img_name=f"{base}_synth{i}.jpg"
            out_json_name=f"{base}_synth{i}.json"
            Image.fromarray(synth_img).save(os.path.join(out_img_dir,out_img_name))
            with open(os.path.join(out_label_dir,out_json_name),"w",encoding="utf-8") as f:
                json.dump({"words":words,"donut_target":donut},f,ensure_ascii=False,indent=2)

    print("Synthetic dataset 생성 완료!")

# -----------------------------
# 실행 예
# -----------------------------
if __name__=="__main__":
    create_synthetic_dataset_from_real(
        input_img_dir=INPUT_IMG_DIR,
        out_img_dir=OUT_IMG_DIR,
        out_label_dir=OUT_LABEL_DIR,
        aug_per_image=AUG_PER_IMAGE
    )
