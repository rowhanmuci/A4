# A4 作業 TODO 代碼片段 - 直接複製使用

"""
這個文件包含所有 TODO 的完整代碼,可以直接複製到對應的 Cell 中。
每個部分都有明確的標記,對應到原 notebook 的位置。
"""

# ============================================================================
# TODO 1: Special Tokens (Cell 17)
# 在 print("After updating special tokens:") 之後插入
# ============================================================================

tokenizer.add_special_tokens({
    'bos_token': '<BOS>',
    'eos_token': '<EOS>',
    'pad_token': '<PAD>'
})


# ============================================================================
# TODO 2: PositionalEncoding (Cell 24)
# 替換整個 PositionalEncoding class 的 __init__ 和 forward 方法
# ============================================================================

# 在 class PositionalEncoding(nn.Module): 之後

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
    """
    B, T, D = x.shape
    return x + self.pe[:, :T, :]


# ============================================================================
# TODO 3: MultiHeadAttention (Cell 24)
# 替換整個 MultiHeadAttention class 的內容
# ============================================================================

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
    return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    B, Tq, _ = Q.shape
    _, Tk, _ = K.shape
    
    Q = self.q_proj(Q)
    K = self.k_proj(K)
    V = self.v_proj(V)
    
    Q = self._shape(Q, B, Tq)
    K = self._shape(K, B, Tk)
    V = self._shape(V, B, Tk)
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
    
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = self.dropout(attn_weights)
    
    attn_output = torch.matmul(attn_weights, V)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
    
    out = self.out_proj(attn_output)
    return out


# ============================================================================
# TODO 4: FeedForward (Cell 24)
# 替換 FeedForward class 的 __init__ 方法
# ============================================================================

def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.0):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(d_model, dim_ff),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_ff, d_model),
        nn.Dropout(dropout)
    )


# ============================================================================
# TODO 5: TransformerDecoderLayer (Cell 24)
# 替換整個 TransformerDecoderLayer class 的內容
# ============================================================================

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


# ============================================================================
# TODO 6: TransformerDecoder (Cell 24)
# 替換整個 TransformerDecoder class 的內容
# ============================================================================

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


# ============================================================================
# TODO 7: generate_greedy (Cell 26)
# 在 CaptionModel class 中的 generate_greedy 方法
# ============================================================================

@torch.no_grad()
def generate_greedy(self, pixel_values: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 32):
    B = pixel_values.shape[0]
    enc = self.encode_image(pixel_values)
    
    generated = torch.full((B, 1), bos_id, dtype=torch.long, device=pixel_values.device)
    
    for _ in range(max_len - 1):
        x = self.token_emb(generated)
        x = self.pos_enc(x)
        
        T = x.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)[None, None, :, :]
        
        dec = self.decoder(x, enc, causal_mask)
        logits = self.lm_head(dec)
        
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        
        if (next_token == eos_id).all():
            break
    
    return generated


# ============================================================================
# TODO 8: Loss and Optimizer (Cell 31)
# ============================================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


# ============================================================================
# TODO 9: Training Loop (Cell 33)
# 替換 train() 函數的內容
# ============================================================================

def train():
    model.train()
    total = 0.0
    
    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        
        logits, x_tgt = model(pixel_values, input_ids)
        
        loss = criterion(logits.reshape(-1, logits.shape[-1]), x_tgt.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total += loss.item()
    
    return total / max(1, len(train_loader))


# ============================================================================
# TODO 10: Evaluation BLEU (Cell 33)
# 在 eval_bleu() 函數中插入
# ============================================================================

bos = tokenizer.bos_token_id
eos = tokenizer.eos_token_id
for batch in loader:
    pix = batch["pixel_values"].to(device)
    
    gen_ids = model.generate_greedy(pix, bos, eos, max_len=GEN_MAX_LEN)
    
    pred_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    preds.extend(pred_texts)
    
    ref_texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    refs.extend([[r] for r in ref_texts])


# ============================================================================
# 額外: 提升分數的技巧
# ============================================================================

# 1. 解凍 ViT 最後幾層 (在 Cell 21 中修改)
"""
vit = ViTModel.from_pretrained(VIT_NAME)
# Freeze all layers first
for p in vit.parameters():
    p.requires_grad = False

# Unfreeze last 2 layers
for name, param in vit.named_parameters():
    if 'encoder.layer.11' in name or 'encoder.layer.10' in name:
        param.requires_grad = True

vit.to(device)
"""

# 2. 添加學習率調度器 (在定義 optimizer 之後)
"""
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
# 或者
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 在每個 epoch 結束後調用:
# scheduler.step()  # for CosineAnnealingLR
# scheduler.step(val_loss)  # for ReduceLROnPlateau
"""

# 3. 梯度累積 (如果記憶體不足)
"""
ACCUMULATION_STEPS = 4  # 將有效 batch size 增加 4 倍

for i, batch in enumerate(train_loader):
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    
    logits, x_tgt = model(pixel_values, input_ids)
    loss = criterion(logits.reshape(-1, logits.shape[-1]), x_tgt.reshape(-1))
    
    loss = loss / ACCUMULATION_STEPS  # 縮放 loss
    loss.backward()
    
    if (i + 1) % ACCUMULATION_STEPS == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
"""

# 4. 改進的超參數設定範例
"""
VIT_NAME = "google/vit-large-patch16-224-in21k"  # 使用更大的模型
TOK_NAME = "gpt2"

MAX_LEN = 40  # 增加最大長度
D_MODEL = 768  # 增加模型維度
N_HEAD = 12  # 增加注意力頭數
FFN_DIM = 3072  # 增加 FFN 維度
N_LAYERS = 6  # 增加層數
DROPOUT = 0.1

LR = 2e-4  # 稍微降低學習率
BATCH_SIZE = 16  # 根據 GPU 記憶體調整
EPOCHS = 10  # 增加訓練輪數

GEN_MAX_LEN = 40
"""
