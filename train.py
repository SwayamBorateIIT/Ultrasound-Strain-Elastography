import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import AdamW
from dataset import BModeDataset
from losses import patch_znssd_loss, smoothness_loss, census_loss
from model import UnDICnet_s
from utils import get_patches, get_strain, warp







transform = transforms.Compose([
  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

root_dir = "/teamspace/studios/this_studio/final_faulty_denoised_png"
dataset = BModeDataset(root_dir=root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)



# Instantiate the model; here args is set to None for simplicity


model = UnDICnet_s(args=None)
device = 'cuda' 

model.to(device)
optimizer = AdamW(model.parameters(), lr=0.0002, weight_decay=0.5e-4)
# Loss weighting parameters as selected in the paper
omega1 = 5
omega2 = 1
model.train()

from tqdm import tqdm
import torch
import os

# Training setup
num_epochs = 300
best_loss = float('inf')
checkpoint_dir = "checkpoints_results_insampled_2sample_smothness3"
os.makedirs(checkpoint_dir, exist_ok=True)

# Learning rate scheduler (e.g., StepLR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

for epoch in range(num_epochs):


    # Create dataset and loader using selected folders

    model.train()
    epoch_loss = 0.0
    loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

    for batch in loop:
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        optimizer.zero_grad()

        outputs = model(img1, img2)
        flow_f = outputs
        img2_warp = warp(img2, flow_f)

        l_sim = patch_znssd_loss(img1, img2_warp)
        l_s = smoothness_loss(flow_f, img1)
        l_c = census_loss(img1, img2_warp)

        loss = l_sim + omega1 * l_s + omega2 * l_c
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    scheduler.step()

    if (epoch ) % 5 == 0:
        plt.imshow(flow_f[0, 1, :, :].cpu().detach().numpy(), cmap='jet')
        plt.colorbar()
        plt.title(f"Flow Y Component at Epoch {epoch+1}")
        plt.show()

        strain_map = get_strain(flow_f[:, 1:2, :, :], 143)  # Use y-displacement
        plt.figure(figsize=(8, 6))
        plt.imshow(strain_map[0, 0].cpu().detach().numpy(), cmap='jet')  # or 'jet' if you prefer
        plt.title("Strain Map (Uxx)")
        # plt.colorbar(label='Strain')
        plt.axis('off')
        plt.show()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss
        }, checkpoint_path)
        print(f" Saved best model at epoch {epoch+1} with loss {best_loss:.6f}")