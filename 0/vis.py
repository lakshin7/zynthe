
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# relative path
folder = "/Users/lakshins/Documents/Zynthe/knowledge-distillation-toolkit/experiments/20250905T195330Z_c5270be8"
#folder = "experiments/20250905T195330Z_c5270be8"

# list images in folder
images = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
img = mpimg.imread(os.path.join(folder, images[1]))

plt.figure(figsize=(10, 10))  # make it big
plt.imshow(img)
plt.title(images[0], fontsize=16)
plt.axis("off")
plt.show()
