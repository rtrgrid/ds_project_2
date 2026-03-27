import imageio
import os

image_folder = "plots/attention"
images = []

files = sorted(os.listdir(image_folder))

for file in files:
    if file.endswith(".png"):
        path = os.path.join(image_folder, file)
        images.append(imageio.imread(path))

# save GIF
# imageio.mimsave("plots/attention/attention.gif", images, duration=0.8)
imageio.mimsave(
    "plots/attention/attention.gif",
    images,
    duration=1.5,   # 🔥 slower (increase more if needed)
    loop=0          # 🔥 infinite loop
)

print("✅ GIF saved")