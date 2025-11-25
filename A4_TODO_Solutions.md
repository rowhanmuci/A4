# A4 作業 TODO 完成指南

這份文件包含了 A4.ipynb 中所有 TODO 項目的完整解決方案。

## 目錄
1. [Special Tokens 設置](#todo-1)
2. [Positional Encoding 實作](#todo-2)
3. [Multi-Head Attention 實作](#todo-3)
4. [FeedForward Network 實作](#todo-4)
5. [Transformer Decoder Layer 實作](#todo-5)
6. [Transformer Decoder 實作](#todo-6)
7. [Greedy Decoding 生成函數](#todo-7)
8. [Loss 和 Optimizer 定義](#todo-8)
9. [Training Loop 實作](#todo-9)
10. [Evaluation BLEU 實作](#todo-10)

---

## TODO 1: Special Tokens 設置 {#todo-1}

**位置**: Cell 17 (在 `print("After updating special tokens:")` 之後)

**要做什麼**: 為 tokenizer 添加 BOS (beginning of sequence), EOS (end of sequence), 和 PAD (padding) 特殊 tokens。

**解決方案**:
```python
# Add special tokens for BOS, EOS, and PAD
tokenizer.add_special_tokens({
    'bos_token': '<BOS>',
    'eos_token': '<EOS>',
    'pad_token': '<PAD>'
})
```

**說明**:
- 使用 `add_special_tokens` 方法將三個特殊 token 加入 tokenizer
- `<BOS>` 標記序列開始
- `<EOS>` 標記序列結束
- `<PAD>` 用於填充序列至相同長度

---

## TODO 2: Positional Encoding 實作 {#todo-2}

**位置**: Cell 24 (PositionalEncoding class)

**要做什麼**: 實作 sinusoidal positional encoding,這是 Transformer 原論文提出的方法。

**解決方案**:
```python
def __init__(self, d_model: int, max_len: int = 512):
    super().__init__()
    # Create positional encoding matrix
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    pe = pe.unsqueeze(0)  # [1, max_len, d_model]
    self.register_buffer('pe', pe)

def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    x: input tensor of shape [batch_size, seq_len, d_model]
    Adds positional encoding to input tensor, up to seq_len.
    """
    B, T, D = x.shape
    return x + self.pe[:, :T, :]
```

**說明**:
- 使用 sine 和 cosine 函數生成位置編碼
- 偶數維度使用 sine,奇數維度使用 cosine
- 使用 `register_buffer` 註冊為非可訓練參數
- 在 forward 中將位置編碼加到輸入 embedding 上

---

## TODO 3: Multi-Head Attention 實作 {#todo-3}

**位置**: Cell 24 (MultiHeadAttention class)

**要做什麼**: 實作 scaled dot-product multi-head attention 機制。

**解決方案**:
```python
def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
    super().__init__()
    assert d_model % num_heads == 0
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads
    
    self.q_proj = nn.Linear(d_model, d_model)
    self.k_proj = nn.Linear(d_model, d_model)
    self.v_proj = nn.Linear(d_model, d_model)
    self.out_proj = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)

def _shape(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """
    x reshape:
    [batch, seq_len, d_model] -> [batch, num_heads, seq_len, head_dim]
    """
    return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    """
    Multi-Head Attention forward pass.
    """
    B, Tq, _ = Q.shape
    _, Tk, _ = K.shape
    
    # Project Q, K, V
    Q = self.q_proj(Q)
    K = self.k_proj(K)
    V = self.v_proj(V)
    
    # Reshape to [B, num_heads, T, head_dim]
    Q = self._shape(Q, B, Tq)
    K = self._shape(K, B, Tk)
    V = self._shape(V, B, Tk)
    
    # Scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
    
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = self.dropout(attn_weights)
    
    # Apply attention to values
    attn_output = torch.matmul(attn_weights, V)
    
    # Reshape back: [B, num_heads, Tq, head_dim] -> [B, Tq, d_model]
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
    
    # Final projection
    out = self.out_proj(attn_output)
    return out
```

**說明**:
- 使用線性層投影 Q, K, V
- 將 d_model 分割成多個 heads (num_heads × head_dim)
- 計算 scaled dot-product attention: `scores = QK^T / sqrt(head_dim)`
- 應用 attention mask (用於 causal attention)
- 將多個 heads 的輸出串接並投影回 d_model

---

## TODO 4: FeedForward Network 實作 {#todo-4}

**位置**: Cell 24 (FeedForward class)

**要做什麼**: 實作 position-wise feedforward network。

**解決方案**:
```python
def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.0):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(d_model, dim_ff),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_ff, d_model),
        nn.Dropout(dropout)
    )
```

**說明**:
- 兩層全連接網路
- 第一層將維度從 d_model 擴展到 dim_ff
- 使用 ReLU 啟動函數
- 第二層將維度投影回 d_model
- 加入 dropout 防止過擬合

---

## TODO 5: Transformer Decoder Layer 實作 {#todo-5}

**位置**: Cell 24 (TransformerDecoderLayer class)

**要做什麼**: 實作單層 Transformer Decoder,包含 masked self-attention, cross-attention 和 FFN。

**解決方案**:
```python
def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float = 0.1):
    super().__init__()
    self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
    self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
    self.ffn = FeedForward(d_model, dim_ff, dropout)
    
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    
    self.dropout = nn.Dropout(dropout)

def forward(self, x, enc, causal_mask: Optional[torch.Tensor] = None):
    """
    Args:
        x: Decoder input embeddings [batch, tgt_seq_len, d_model]
        enc: Encoder output representations [batch, src_seq_len, d_model]
        causal_mask: Optional attention mask for target-side self-attention
    """
    # Masked self-attention
    x_norm = self.norm1(x)
    attn_out = self.self_attn(x_norm, x_norm, x_norm, causal_mask)
    x = x + self.dropout(attn_out)
    
    # Cross-attention
    x_norm = self.norm2(x)
    attn_out = self.cross_attn(x_norm, enc, enc)
    x = x + self.dropout(attn_out)
    
    # Feed-forward
    x_norm = self.norm3(x)
    ffn_out = self.ffn(x_norm)
    x = x + self.dropout(ffn_out)
    
    return x
```

**說明**:
- 包含三個主要組件:
  1. **Masked Self-Attention**: decoder 對自身的 attention (帶 causal mask)
  2. **Cross-Attention**: decoder 對 encoder 輸出的 attention
  3. **Feed-Forward Network**: position-wise FFN
- 每個組件都使用 residual connection 和 Layer Normalization
- 使用 Pre-LN 架構 (先 normalize 再做 attention/FFN)

---

## TODO 6: Transformer Decoder 實作 {#todo-6}

**位置**: Cell 24 (TransformerDecoder class)

**要做什麼**: 堆疊多層 TransformerDecoderLayer。

**解決方案**:
```python
def __init__(self, d_model: int, num_layers: int, num_heads: int, dim_ff: int, dropout: float = 0.1):
    super().__init__()
    self.layers = nn.ModuleList([
        TransformerDecoderLayer(d_model, num_heads, dim_ff, dropout)
        for _ in range(num_layers)
    ])
    self.ln = nn.LayerNorm(d_model)

def forward(self, x, enc, causal_mask: Optional[torch.Tensor] = None):
    for layer in self.layers:
        x = layer(x, enc, causal_mask)
    return self.ln(x)
```

**說明**:
- 使用 `nn.ModuleList` 創建多層 decoder layers
- 依序通過每一層
- 最後加上 LayerNorm 穩定輸出

---

## TODO 7: Greedy Decoding 生成函數 {#todo-7}

**位置**: Cell 26 (CaptionModel.generate_greedy method)

**要做什麼**: 實作 greedy decoding 算法來生成 image captions。

**解決方案**:
```python
def generate_greedy(self, pixel_values: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 32):
    """
    Greedy decoding for image-to-text generation.
    """
    B = pixel_values.shape[0]
    enc = self.encode_image(pixel_values)
    
    # Start with BOS token
    generated = torch.full((B, 1), bos_id, dtype=torch.long, device=pixel_values.device)
    
    for _ in range(max_len - 1):
        # Embed and add positional encoding
        x = self.token_emb(generated)
        x = self.pos_enc(x)
        
        # Create causal mask
        T = x.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)[None, None, :, :]
        
        # Decode
        dec = self.decoder(x, enc, causal_mask)
        logits = self.lm_head(dec)
        
        # Get next token (greedy)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        
        # Check if all sequences have generated EOS
        if (next_token == eos_id).all():
            break
    
    return generated
```

**說明**:
- 從 BOS token 開始
- 每步選擇機率最高的 token (greedy)
- 自回歸生成:每次將新 token 加入序列
- 當所有序列都生成 EOS 或達到 max_len 時停止

---

## TODO 8: Loss 和 Optimizer 定義 {#todo-8}

**位置**: Cell 31

**要做什麼**: 定義損失函數和優化器。

**解決方案**:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
```

**說明**:
- 使用 AdamW optimizer (Adam with weight decay)
- 使用 CrossEntropyLoss
- `ignore_index=tokenizer.pad_token_id` 確保 padding tokens 不參與 loss 計算

---

## TODO 9: Training Loop 實作 {#todo-9}

**位置**: Cell 33 (train function)

**要做什麼**: 實作完整的訓練循環。

**解決方案**:
```python
def train():
    model.train()
    total = 0.0
    
    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        
        # Forward pass
        logits, x_tgt = model(pixel_values, input_ids)
        
        # Compute loss
        loss = criterion(logits.reshape(-1, logits.shape[-1]), x_tgt.reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        total += loss.item()
    
    return total / max(1, len(train_loader))
```

**說明**:
- 設置模型為訓練模式
- 遍歷所有 training batches
- Forward pass 計算 logits
- 計算 cross-entropy loss
- Backward pass 計算梯度
- 使用 gradient clipping 防止梯度爆炸
- 更新模型參數
- 返回平均 loss

---

## TODO 10: Evaluation BLEU 實作 {#todo-10}

**位置**: Cell 33 (eval_bleu function 內部)

**要做什麼**: 實作評估循環,使用模型生成 captions 並計算 BLEU 分數。

**解決方案**:
```python
bos = tokenizer.bos_token_id
eos = tokenizer.eos_token_id
for batch in loader:
    pix = batch["pixel_values"].to(device)
    
    # Generate captions
    gen_ids = model.generate_greedy(pix, bos, eos, max_len=GEN_MAX_LEN)
    
    # Decode predictions
    pred_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    preds.extend(pred_texts)
    
    # Get reference captions
    ref_texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    refs.extend([[r] for r in ref_texts])
```

**說明**:
- 獲取 BOS 和 EOS token IDs
- 遍歷 validation/test loader
- 使用 `generate_greedy` 生成預測的 captions
- 使用 tokenizer 將 token IDs 解碼為文字
- 收集預測和參考 captions
- 注意 refs 需要是 list of lists 格式 (每個樣本可能有多個參考 captions)

---

## 額外建議

### 提升分數的策略:

1. **解凍部分 ViT 層**:
   - 在 Cell 21 中,可以嘗試解凍 ViT 的最後幾層進行 fine-tuning
   ```python
   # Unfreeze last few layers
   for name, param in vit.named_parameters():
       if 'encoder.layer.11' in name or 'encoder.layer.10' in name:
           param.requires_grad = True
   ```

2. **調整超參數**:
   - 增加 `EPOCHS` (例如 10-15)
   - 調整學習率 `LR` (嘗試 1e-4 到 5e-4)
   - 增加模型容量:`D_MODEL`, `N_LAYERS`, `FFN_DIM`
   - 調整 `BATCH_SIZE` 以適應 GPU 記憶體

3. **使用學習率調度器**:
   ```python
   from torch.optim.lr_scheduler import CosineAnnealingLR
   scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
   ```

4. **資料增強**:
   - 在 image preprocessing 中加入資料增強

5. **使用更好的 decoding 策略**:
   - 實作 beam search 而非 greedy decoding
   - 使用 nucleus sampling (top-p sampling)

### 常見問題:

1. **CUDA out of memory**:
   - 減小 `BATCH_SIZE`
   - 減小模型大小 (`D_MODEL`, `N_LAYERS`)
   - 使用梯度累積

2. **訓練不穩定**:
   - 檢查 gradient clipping 設定
   - 降低學習率
   - 增加 warmup steps

3. **BLEU 分數過低**:
   - 檢查 tokenizer 設定
   - 確保 special tokens 正確處理
   - 增加訓練 epochs

---

## 完成檢查清單

- [ ] TODO 1: Special tokens 設置完成
- [ ] TODO 2: PositionalEncoding 實作完成
- [ ] TODO 3: MultiHeadAttention 實作完成
- [ ] TODO 4: FeedForward 實作完成
- [ ] TODO 5: TransformerDecoderLayer 實作完成
- [ ] TODO 6: TransformerDecoder 實作完成
- [ ] TODO 7: generate_greedy 實作完成
- [ ] TODO 8: Loss 和 Optimizer 定義完成
- [ ] TODO 9: Training loop 實作完成
- [ ] TODO 10: Evaluation 實作完成
- [ ] 填寫姓名和學號
- [ ] 完成報告 (10分)
- [ ] 訓練模型並提交測試結果

祝你作業順利! 🎉
