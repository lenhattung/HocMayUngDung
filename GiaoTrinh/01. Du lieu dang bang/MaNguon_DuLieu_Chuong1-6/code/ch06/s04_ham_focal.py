# Ô lệnh 4 -- Viết focal loss bằng PyTorch và kiểm tra với trường hợp gamma = 0
def focal_loss(logits, y, gamma=2.0, alpha=None):
    """FL = -alpha_t * (1 - p_t)^gamma * log(p_t), lấy trung bình trên các mẫu."""
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")   # = -log(p_t), ổn định
    p = torch.sigmoid(logits)
    p_t = p * y + (1 - p) * (1 - y)
    loss = (1 - p_t) ** gamma * ce
    if alpha is not None:                       # trọng số lớp: alpha cho lớp 1, 1 - alpha cho lớp 0
        loss = (alpha * y + (1 - alpha) * (1 - y)) * loss
    return loss.mean()

z = torch.tensor([2.0, -0.5, -2.0, 0.8])        # "logit": điểm số trước khi qua sigmoid
yt = torch.tensor([1.0, 1.0, 0.0, 0.0])
print("BCE của PyTorch       :", round(F.binary_cross_entropy_with_logits(z, yt).item(), 4))
print("focal_loss, gamma = 0 :", round(focal_loss(z, yt, gamma=0).item(), 4))
print("focal_loss, gamma = 2 :", round(focal_loss(z, yt, gamma=2).item(), 4))
