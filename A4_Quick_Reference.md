# A4 作業 - 快速參考卡片 🚀

## 📍 TODO 位置速查表

| TODO | Cell | 位置標記 | 難度 | 分數 |
|------|------|---------|------|------|
| 1. Special Tokens | 17 | `print("After updating...")` 後 | ⭐ | 必須 |
| 2. Positional Encoding | 24 | `class PositionalEncoding` | ⭐⭐ | 5 |
| 3. Multi-Head Attention | 24 | `class MultiHeadAttention` | ⭐⭐⭐ | 10 |
| 4. Feed-Forward | 24 | `class FeedForward` | ⭐ | 5 |
| 5. Decoder Layer | 24 | `class TransformerDecoderLayer` | ⭐⭐⭐ | 8 |
| 6. Decoder | 24 | `class TransformerDecoder` | ⭐⭐ | 5 |
| 7. Generate Greedy | 26 | `def generate_greedy` | ⭐⭐⭐ | 10 |
| 8. Loss & Optimizer | 31 | `# TODO: Define loss...` | ⭐ | 必須 |
| 9. Training Loop | 33 | `def train()` | ⭐⭐⭐ | 15 |
| 10. Evaluation | 33 | `def eval_bleu()` | ⭐⭐ | 15 |

難度: ⭐ 簡單 | ⭐⭐ 中等 | ⭐⭐⭐ 困難

---

## ⚡ 超快速實作順序

### Phase 1: 基礎組件 (30 分鐘)
1. TODO 1: Special tokens → 2 行代碼
2. TODO 4: FeedForward → 簡單的 Sequential
3. TODO 8: Loss & Optimizer → 2 行代碼

### Phase 2: Attention 機制 (45 分鐘)
4. TODO 2: Positional Encoding → 數學公式實作
5. TODO 3: Multi-Head Attention → 核心機制

### Phase 3: Decoder (30 分鐘)
6. TODO 5: Decoder Layer → 組裝 attention + FFN
7. TODO 6: Decoder → 堆疊 layers

### Phase 4: 訓練與生成 (45 分鐘)
8. TODO 7: Generate function → 自回歸生成
9. TODO 9: Training loop → 完整訓練流程
10. TODO 10: Evaluation → BLEU 計算

**總時間**: 約 2.5 小時

---

## 🔑 關鍵代碼速記

### 1. Special Tokens (1 行)
```python
tokenizer.add_special_tokens({'bos_token':'<BOS>','eos_token':'<EOS>','pad_token':'<PAD>'})
```

### 2. Positional Encoding 核心
```python
pe[:, 0::2] = torch.sin(position * div_term)  # 偶數
pe[:, 1::2] = torch.cos(position * div_term)  # 奇數
```

### 3. Attention 核心計算
```python
scores = Q @ K.T / sqrt(head_dim)
attn = softmax(masked_fill(scores))
output = attn @ V
```

### 4. Decoder Layer 順序
```python
x + dropout(self_attn(norm(x)))      # 1. Masked Self-Attn
x + dropout(cross_attn(norm(x), enc)) # 2. Cross-Attn
x + dropout(ffn(norm(x)))             # 3. FFN
```

### 5. Greedy Decoding 核心
```python
for _ in range(max_len):
    logits = model(generated)
    next_token = logits[:,-1,:].argmax(dim=-1)
    generated = cat([generated, next_token])
```

---

## 🐛 Debug 速查

### 問題 1: Shape 不匹配
```python
# 檢查維度
print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
print(f"Expected: [B, T, d_model]")
```

### 問題 2: Attention mask 錯誤
```python
# Causal mask 應該是上三角
mask = torch.triu(torch.ones(T,T), diagonal=1).bool()
# True 的位置會被 mask 掉
```

### 問題 3: Loss 是 NaN
```python
# 檢查:
1. 學習率太大? → 降到 1e-4
2. Gradient exploding? → 確認有 clip_grad_norm
3. PAD token? → ignore_index=pad_token_id
```

### 問題 4: CUDA OOM
```python
# 解決方案:
BATCH_SIZE = 16  # 減小
D_MODEL = 512    # 減小
N_LAYERS = 4     # 減小
```

---

## 📐 維度速查表

```
Image:            [B, 3, 224, 224]
                     ↓ ViT
ViT Output:       [B, 197, 1024]  (197 = 196 patches + 1 CLS)
                     ↓ Linear Proj
Encoder Output:   [B, 197, d_model]

Input IDs:        [B, T]
                     ↓ Embedding
Token Embeddings: [B, T, d_model]
                     ↓ + PE
With PE:          [B, T, d_model]
                     ↓ Decoder
Decoder Output:   [B, T, d_model]
                     ↓ LM Head
Logits:           [B, T, vocab_size]

Loss Input:       [B*T, vocab_size] vs [B*T]
```

---

## ⚙️ 超參數速配指南

### 標準配置 (可以跑)
```python
D_MODEL = 512
N_HEAD = 8
FFN_DIM = 2048
N_LAYERS = 4
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 5
```

### 高分配置 (需要好 GPU)
```python
D_MODEL = 768
N_HEAD = 12
FFN_DIM = 3072
N_LAYERS = 6
BATCH_SIZE = 16
LR = 2e-4
EPOCHS = 10
```

### 快速測試配置
```python
D_MODEL = 256
N_HEAD = 4
FFN_DIM = 1024
N_LAYERS = 2
BATCH_SIZE = 8
LR = 5e-4
EPOCHS = 2
```

---

## 💻 必備 Import

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ViTModel, AutoImageProcessor, AutoTokenizer
from datasets import load_dataset
import evaluate
import math
from typing import Optional
```

---

## 🎯 提分技巧排序 (按效果)

1. **增加 Epochs** (10→15) → 🔥🔥🔥 效果最好
2. **Fine-tune ViT 最後 2 層** → 🔥🔥🔥
3. **使用 Learning Rate Scheduler** → 🔥🔥
4. **增加模型大小** (D_MODEL, N_LAYERS) → 🔥🔥
5. **調整 LR** → 🔥
6. **實作 Beam Search** → 🔥🔥 (但需要額外實作)
7. **資料增強** → 🔥

---

## 🔧 一鍵優化代碼

### 1. Fine-tune ViT
```python
# 解凍最後 2 層
for name, param in vit.named_parameters():
    if 'layer.11' in name or 'layer.10' in name:
        param.requires_grad = True
```

### 2. Learning Rate Scheduler
```python
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
# 每個 epoch 後: scheduler.step()
```

### 3. 監控訓練
```python
from tqdm import tqdm
for epoch in range(EPOCHS):
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch in pbar:
        # training code...
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
```

---

## 📊 預期結果

### 訓練過程
```
Epoch 1: Train Loss ~3.5, Val BLEU ~10
Epoch 3: Train Loss ~2.0, Val BLEU ~20
Epoch 5: Train Loss ~1.5, Val BLEU ~25-30
Epoch 10: Train Loss ~1.0, Val BLEU ~30-35
```

### 合格分數
- **基礎分**: BLEU > 15 (完成所有 TODO)
- **中等分**: BLEU > 25 (調參 + 多訓練)
- **高分**: BLEU > 30 (Fine-tune + 優化)
- **頂尖**: BLEU > 35 (全部優化)

---

## ⏱️ 時間管理建議

### Day 1 (2 小時)
- 完成 TODO 1-8
- 跑通整個 pipeline
- 測試是否能訓練

### Day 2 (3 小時)
- 完成 TODO 9-10
- 訓練初版模型
- 得到基礎結果

### Day 3 (2 小時)
- 調整超參數
- 優化模型
- 多次訓練提升分數

### Day 4 (1 小時)
- 撰寫報告
- 整理代碼
- 最終檢查

**總時間**: 約 8 小時

---

## 📝 報告寫作要點

### 必須包含
1. **實作說明** (30%)
   - 每個 TODO 做了什麼
   - 關鍵設計選擇

2. **遇到的問題** (30%)
   - 列舉 2-3 個問題
   - 如何解決

3. **實驗結果** (30%)
   - 訓練曲線
   - BLEU 分數
   - 生成範例

4. **總結與反思** (10%)
   - 學到什麼
   - 可以改進的地方

---

## 🚨 最後檢查清單

提交前確認:
- [ ] 所有 TODO 都已完成
- [ ] 程式可以完整執行
- [ ] 已訓練並得到 BLEU 分數
- [ ] 姓名學號已填寫
- [ ] 報告已完成
- [ ] 所有 cell 都執行過
- [ ] 儲存了 notebook

---

## 🎓 評分計算器

```
總分 = TODO 實作 (80) + 報告 (10) + 加分 (0-10)

你的目標:
- 基本及格: 60 分 (完成所有 TODO)
- 良好: 75 分 (TODO + 好的報告)
- 優秀: 85+ 分 (TODO + 報告 + 高 BLEU)
- 滿分: 95-100 分 (全部做好 + Top 10% BLEU)
```

---

## 💡 最後提醒

1. **先求有,再求好** - 先讓代碼跑起來
2. **邊做邊測試** - 不要寫完才測試
3. **善用 print()** - Debug 的好朋友
4. **參考文檔** - 查看已提供的完整解決方案
5. **時間管理** - 不要拖到最後一天

---

## 📞 遇到問題?

參考這些文件:
1. `A4_TODO_Solutions.md` - 詳細解決方案
2. `A4_Code_Snippets.py` - 可直接複製的代碼
3. `A4_Architecture_Guide.md` - 架構說明

實在卡住了:
- 重新閱讀題目要求
- 檢查維度是否正確
- 參考 PyTorch 官方文檔
- 查看 error message 詳細內容

---

**祝你拿高分! 💯**

記住: Code first, optimize later! 🚀
