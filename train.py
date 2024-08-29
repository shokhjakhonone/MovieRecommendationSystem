import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Загрузка данных
url = "https://datasets.imdbws.com/title.ratings.tsv.gz"
data = pd.read_csv(url, sep='\t')
data.dropna(inplace=True)
data['tconst'] = data['tconst'].astype('category').cat.codes
data['averageRating'] = data['averageRating'].astype(float)
data = data.rename(columns={'tconst': 'item_id', 'rating': 'averageRating'})
data['user_id'] = pd.factorize(data['item_id'])[0]

# Параметры модели
n_users = data['user_id'].nunique()
n_items = data['item_id'].nunique()
embedding_size = 50

# Создание PyTorch Dataset
class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['averageRating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

dataset = RatingsDataset(data)
train_loader = DataLoader(dataset, batch_size=512, shuffle=True)

# Построение модели
class RecommenderNet(nn.Module):
    def __init__(self, n_users, n_items, embedding_size):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.item_embedding = nn.Embedding(n_items, embedding_size)
        self.fc = nn.Linear(embedding_size * 2, 1)
    
    def forward(self, users, items):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        x = torch.cat([user_emb, item_emb], dim=1)
        x = self.fc(x)
        return x

model = RecommenderNet(n_users=n_users, n_items=n_items, embedding_size=embedding_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Перенос модели и данных на GPU, если доступен
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion.to(device)

# Увеличение размера пакета
train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
pretrained_embeddings = {
    'user_embedding': model.user_embedding.weight.data,
    'item_embedding': model.item_embedding.weight.data
}
torch.save(pretrained_embeddings, 'pretrained_embeddings.pt')

# Использование предобученных эмбеддингов
pretrained_embeddings = torch.load('pretrained_embeddings.pt')
model.user_embedding.weight.data.copy_(pretrained_embeddings['user_embedding'])
model.item_embedding.weight.data.copy_(pretrained_embeddings['item_embedding'])

# Использование более эффективного оптимизатора
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Увеличение количества эпох обучения
n_epochs = 10

# Обучение модели на GPU
model.train()
for epoch in range(n_epochs):
    total_loss = 0
    for batch_idx, (users, items, ratings) in enumerate(train_loader):
        users = users.to(device)
        items = items.to(device)
        ratings = ratings.to(device)

        optimizer.zero_grad()
        predictions = model(users, items).squeeze()
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}")

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# Сохранение модели
torch.save(model.state_dict(), 'model.pt')
