from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.network = nn.Sepuential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits
    
def train(model,dataloader,loss_fn,optimizer):

    model.train()
    for image_batch,label_batch in dataloader:

        logits_batch = model(image_batch)

        loss=loss_fn(logits_batch,label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.list()

acc_test=models.test_accuracy(model,dataloader_test)
print(f'test accuracy: {acc_test*100:.2f}%')

n_epochs=5

for k in range(n_epochs):
    print(f'epoch {k+1}/{n_epochs}',end=': ',flush=True)

    loss_train=models.train(model,dataloader_train,loss_fn,optimizer)
    print(f'train loss:{loss_train}')
    acc_test=models.test_accuracy(model,dataloader_test)
    print(f'test accuracy:{acc_test*100:.2f}%')

def test(model, dataloader, loss_fn):
    loss_total = 0.0  

    model.eval()  
    for image_batch, label_batch in dataloader:  
        with torch.no_grad():  
            logits_batch = model(image_batch)  

            loss = loss_fn(logits_batch, label_batch)
            loss_total += loss.item()  

    return loss_total / len(dataloader)

