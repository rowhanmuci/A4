# A4 作業架構總覽與實作指南

## 📋 目錄
1. [整體架構](#architecture)
2. [各組件詳解](#components)
3. [實作步驟](#steps)
4. [常見問題](#faq)
5. [評分標準](#grading)

---

## 🏗️ 整體架構 {#architecture}

```
Image Captioning Model 架構

                    Input Image
                         |
                         v
              +-------------------+
              | Vision Transformer |  ← 預訓練 ViT (凍結/部分解凍)
              | (Image Encoder)    |
              +-------------------+
                         |
                    Visual Features
                    [B, 197, 1024]
                         |
                         v
              +-------------------+
              |  Linear Projection |  ← 投影到 d_model
              +-------------------+
                         |
                    Encoder Output
                    [B, 197, 512]
                         |
                         v
    ┌────────────────────┴─────────────────────┐
    |                                           |
    |         Transformer Decoder               |
    |  ┌─────────────────────────────────┐    |
    |  │  Token Embedding                 │    |
    |  │  + Positional Encoding           │    |
    |  └─────────────────────────────────┘    |
    |                 |                         |
    |  ┌─────────────v──────────────────┐     |
    |  │ Decoder Layer 1                 │     |
    |  │  • Masked Self-Attention        │     |
    |  │  • Cross-Attention              │     |
    |  │  • Feed-Forward Network         │     |
    |  └─────────────┬──────────────────┘     |
    |                |                         |
    |  ┌─────────────v──────────────────┐     |
    |  │ Decoder Layer 2-N               │     |
    |  │  • ...                          │     |
    |  └─────────────┬──────────────────┘     |
    |                |                         |
    |  ┌─────────────v──────────────────┐     |
    |  │ Layer Normalization             │     |
    |  └─────────────────────────────────┘    |
    |                                           |
    └───────────────────┬───────────────────────┘
                        |
                        v
              +-------------------+
              | Language Model    |  ← Linear layer to vocab
              | Head              |
              +-------------------+
                        |
                        v
                   Output Logits
                   [B, T, vocab_size]
                        |
                        v
                  Generated Caption
```

---

## 🔧 各組件詳解 {#components}

### 1️⃣ Positional Encoding

**作用**: 為 token 序列注入位置資訊

**公式**:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**關鍵點**:
- 偶數維度使用 sine,奇數維度使用 cosine
- 使用 `register_buffer` 註冊為非訓練參數
- 在 forward 時加到 input embeddings 上

---

### 2️⃣ Multi-Head Attention

**作用**: 讓模型關注序列中不同位置的資訊

**流程**:
```
Input: Q, K, V [batch, seq_len, d_model]
  |
  v
Linear Projections (Q_proj, K_proj, V_proj)
  |
  v
Split into H heads: [batch, num_heads, seq_len, head_dim]
  |
  v
Scaled Dot-Product Attention:
  scores = (Q @ K^T) / sqrt(head_dim)
  attn = softmax(masked_fill(scores))
  output = attn @ V
  |
  v
Concatenate heads: [batch, seq_len, d_model]
  |
  v
Output Projection
```

**關鍵參數**:
- `d_model`: 模型維度 (512)
- `num_heads`: 注意力頭數 (8)
- `head_dim = d_model // num_heads` (64)

---

### 3️⃣ Feed-Forward Network

**結構**:
```
Input [B, T, d_model]
  |
  v
Linear(d_model → dim_ff)  # 擴展
  |
  v
ReLU()
  |
  v
Dropout()
  |
  v
Linear(dim_ff → d_model)  # 壓縮
  |
  v
Dropout()
  |
  v
Output [B, T, d_model]
```

**作用**: 對每個位置獨立進行非線性變換

---

### 4️⃣ Transformer Decoder Layer

**組成**:
```
Input x [B, T, d_model]
  |
  ├─> LayerNorm ─> Masked Self-Attention ──┐
  |                                         |
  └─────────────────── + ──────────────────┘
  |
  ├─> LayerNorm ─> Cross-Attention ────────┐
  |                (attend to encoder)     |
  └─────────────────── + ──────────────────┘
  |
  ├─> LayerNorm ─> Feed-Forward ───────────┐
  |                                         |
  └─────────────────── + ──────────────────┘
  |
  v
Output x [B, T, d_model]
```

**三個主要操作**:
1. **Masked Self-Attention**: 防止看到未來的 tokens
2. **Cross-Attention**: 將圖像特徵融入到文字生成中
3. **Feed-Forward**: 位置獨立的非線性變換

---

### 5️⃣ Greedy Decoding

**生成流程**:
```
1. 初始化: generated = [BOS]
2. Loop (max_len 次):
   a. 將 generated tokens 轉為 embeddings
   b. 加上 positional encoding
   c. 通過 decoder (使用 causal mask)
   d. 取最後一個位置的 logits
   e. 選擇機率最高的 token: next_token = argmax(logits[-1])
   f. 將 next_token 加入 generated
   g. 如果 next_token == EOS,停止
3. 返回: generated caption
```

**Causal Mask**:
```
在時間步 t,只能看到 t 及之前的 tokens
Mask matrix (上三角為 True):
    t0  t1  t2  t3
t0  0   1   1   1
t1  0   0   1   1
t2  0   0   0   1
t3  0   0   0   0
```

---

## 📝 實作步驟 {#steps}

### Step 1: 準備環境 ✅
- 安裝必要套件
- 載入 dataset
- 設定 device

### Step 2: 設定 Special Tokens ✅
```python
tokenizer.add_special_tokens({
    'bos_token': '<BOS>',
    'eos_token': '<EOS>',
    'pad_token': '<PAD>'
})
```

### Step 3: 實作 Positional Encoding ✅
- 計算 sinusoidal encoding
- 註冊為 buffer
- 在 forward 中加到 input

### Step 4: 實作 Multi-Head Attention ✅
- Q, K, V 投影
- 分割成多個 heads
- Scaled dot-product attention
- 串接並投影回 d_model

### Step 5: 實作 Feed-Forward Network ✅
- 兩層線性層
- 中間加 ReLU 和 Dropout

### Step 6: 實作 Decoder Layer ✅
- Masked self-attention
- Cross-attention
- Feed-forward
- 殘差連接 + LayerNorm

### Step 7: 組裝完整 Decoder ✅
- 堆疊多層 decoder layers
- 最後加 LayerNorm

### Step 8: 實作生成函數 ✅
- Greedy decoding
- 自回歸生成
- 使用 causal mask

### Step 9: 定義訓練設定 ✅
- Loss: CrossEntropyLoss
- Optimizer: AdamW
- Learning rate

### Step 10: 實作訓練和評估 ✅
- Training loop
- Gradient clipping
- BLEU evaluation

---

## ❓ 常見問題 {#faq}

### Q1: CUDA out of memory 怎麼辦?
**A**: 
1. 減小 `BATCH_SIZE` (例如從 32 → 16)
2. 減小模型大小 (`D_MODEL`, `N_LAYERS`)
3. 使用 gradient accumulation
4. 在 Colab 中: Runtime → Factory reset runtime

### Q2: 訓練很慢怎麼辦?
**A**:
1. 確保使用 GPU (檢查 `device`)
2. 減小 `MAX_LEN`
3. 凍結 ViT (不要 fine-tune)
4. 使用更小的 ViT 模型

### Q3: BLEU 分數很低怎麼辦?
**A**:
1. 檢查 special tokens 是否正確設定
2. 增加訓練 epochs
3. 調整學習率
4. 檢查 generate_greedy 實作
5. 確保 eval 時正確使用 BOS/EOS

### Q4: Loss 不下降或 NaN?
**A**:
1. 降低學習率
2. 檢查 gradient clipping
3. 檢查 attention mask 是否正確
4. 確保 PAD token 在 loss 中被 ignore

### Q5: 如何提升分數?
**A**:
1. 解凍 ViT 最後幾層 fine-tune
2. 增加訓練 epochs (10-15)
3. 使用 learning rate scheduler
4. 增加模型容量 (D_MODEL, N_LAYERS)
5. 實作 beam search 代替 greedy
6. 資料增強

---

## 📊 評分標準 {#grading}

### TODO 實作: 80 分
- TODO 1: Special tokens (必須)
- TODO 2: Positional Encoding (5 分)
- TODO 3: Multi-Head Attention (10 分)
- TODO 4: Feed-Forward (5 分)
- TODO 5: Decoder Layer (8 分)
- TODO 6: Decoder (5 分)
- TODO 7: Generate function (10 分)
- TODO 8: Loss & Optimizer (必須)
- TODO 9: Training loop (15 分)
- TODO 10: Evaluation (15 分)
- 其他: Setup, Model assembly (7 分)

### 報告: 10 分
需包含:
- 實作說明
- 設計選擇 (為什麼這樣設計)
- 遇到的問題與解決方式
- 訓練結果分析

### 加分題: 10 分
根據 test set 上的 sacreBLEU 分數排名:
- Top 10%: +10 分
- 10-30%: +8 分
- 30-50%: +6 分
- 50-70%: +4 分
- 70-90%: +2 分
- Bottom 10%: +0 分

---

## 🎯 實作檢查清單

### 必做項目
- [ ] 填寫姓名和學號
- [ ] TODO 1-10 全部完成
- [ ] 程式能成功執行
- [ ] 完成訓練並得到結果
- [ ] 撰寫報告

### 提升分數 (選做)
- [ ] Fine-tune ViT 部分層
- [ ] 使用 learning rate scheduler
- [ ] 調整超參數獲得更好結果
- [ ] 實作 beam search
- [ ] 加入資料增強

---

## 💡 實用技巧

### 1. 快速測試
```python
# 用小 batch 先測試整個 pipeline
test_batch = next(iter(train_loader))
logits, x_tgt = model(
    test_batch["pixel_values"][:2].to(device),
    test_batch["input_ids"][:2].to(device)
)
print(f"Logits shape: {logits.shape}")
print(f"Target shape: {x_tgt.shape}")
```

### 2. 監控訓練
```python
# 在 training loop 中加入
if (i + 1) % 10 == 0:
    print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
```

### 3. 儲存模型
```python
# 在每個 epoch 結束後
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_loss,
}, f'checkpoint_epoch_{epoch}.pt')
```

### 4. 視覺化生成結果
```python
# 隨機選幾張圖看看生成的 caption
import random
idx = random.randint(0, len(test_loader.dataset))
sample = test_loader.dataset[idx]
image = sample['image']
generated = model.generate_greedy(
    image_processor(image, return_tensors='pt')['pixel_values'].to(device),
    tokenizer.bos_token_id,
    tokenizer.eos_token_id
)
caption = tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"Generated: {caption}")
```

---

## 📚 參考資源

1. **Transformer 原論文**: "Attention Is All You Need"
2. **ViT 論文**: "An Image is Worth 16x16 Words"
3. **Hugging Face Transformers 文檔**
4. **PyTorch 官方教學**

---

## 🎉 祝你作業順利!

記得:
1. 先完成所有 TODO
2. 確保程式能跑
3. 調整參數提升分數
4. 撰寫報告說明你的實作

有問題可以參考:
- `A4_TODO_Solutions.md` - 詳細的解決方案說明
- `A4_Code_Snippets.py` - 可直接複製的代碼片段

Good luck! 🚀
