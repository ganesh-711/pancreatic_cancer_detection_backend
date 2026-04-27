import os
import shutil
import random

# Your dataset paths
base_train = "C:/Users/sivak/OneDrive/Desktop/PancreasCancerDetection/dataset/train"
base_val = "C:/Users/sivak/OneDrive/Desktop/PancreasCancerDetection/dataset/val"
base_test = "C:/Users/sivak/OneDrive/Desktop/PancreasCancerDetection/dataset/test"

classes = ["cancer", "normal"]

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

def clear_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        for item in os.listdir(folder):
            path = os.path.join(folder, item)
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)

# Clear val and test before splitting
for cls in classes:
    clear_folder(os.path.join(base_val, cls))
    clear_folder(os.path.join(base_test, cls))

# Split data
for cls in classes:
    folder = os.path.join(base_train, cls)
    images = os.listdir(folder)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for img in val_images:
        shutil.move(os.path.join(folder, img), os.path.join(base_val, cls, img))

    for img in test_images:
        shutil.move(os.path.join(folder, img), os.path.join(base_test, cls, img))

print("✅ Split complete! Train / Val / Test folders are ready.")
