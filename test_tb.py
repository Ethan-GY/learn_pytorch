import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
image_path = "dataset/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
writer.add_image("test",img_array,dataformats="HWC")
#y = x
for i in range(100):
    writer.add_scalar("y=x", i, i)


writer.close()