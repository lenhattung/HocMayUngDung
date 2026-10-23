# Ô lệnh 5 -- Mạng nơ-ron nhiều tầng (MLP) và hàm huấn luyện
class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)          # trả về logit, mỗi mẫu một số

def train_mlp(X, y, loss_fn, epochs=25, batch=256, lr=1e-3, seed=SEED, verbose=False):
    torch.manual_seed(seed)
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y), dtype=torch.float32)
    model = MLP(Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))          # xáo trộn dữ liệu mỗi epoch
        for i in range(0, len(Xt), batch):      # chia thành các lô (mini-batch) 256 mẫu
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()                     # tính đạo hàm
            opt.step()                          # cập nhật trọng số
        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: mất mát của lô cuối = {loss.item():.4f}")
    model.eval()
    return model

@torch.no_grad()
def predict_proba(model, X):
    return torch.sigmoid(model(torch.tensor(np.asarray(X), dtype=torch.float32))).numpy()

bce = lambda z, y: F.binary_cross_entropy_with_logits(z, y)
mlp_bce = train_mlp(Xtr, y_train, bce, verbose=True)
