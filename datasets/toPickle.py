# import os
# import pickle
# from PIL import Image
# import numpy as np
#
# def convert_split(split_dir, split_name, save_dir):
#     """
#     Converts a directory of class folders into a single pickle file
#     with keys: 'data' and 'labels', as required by PyTorch-MAML.
#     """
#     all_images = []
#     all_labels = []
#     class_folders = sorted(os.listdir(split_dir))
#     print(f"[{split_name}] Found {len(class_folders)} classes.")
#
#     for idx, cls in enumerate(class_folders):
#         cls_path = os.path.join(split_dir, cls)
#         if not os.path.isdir(cls_path):
#             continue
#         imgs = os.listdir(cls_path)
#         for img_name in imgs:
#             img_path = os.path.join(cls_path, img_name)
#             try:
#                 img = Image.open(img_path).convert("RGB").resize((84, 84))
#                 all_images.append(np.array(img))
#                 all_labels.append(idx)
#             except Exception as e:
#                 print(f"⚠️ Skipping {img_path}: {e}")
#
#     pack = {
#         "data": np.array(all_images, dtype=np.uint8),
#         "labels": np.array(all_labels, dtype=np.int64),
#     }
#
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"train_phase_{split_name}.pickle")
#     with open(save_path, "wb") as f:
#         pickle.dump(pack, f)
#     print(f"✅ Saved {save_path} ({len(all_labels)} samples)")
#
# if __name__ == "__main__":
#     # klasörleri kendi sistemine göre ayarla:
#     base_dir = r"C:\Users\aligunduz\PycharmProjects\PyTorch-MAML\data\omniglot"
#     save_dir = r"C:\Users\aligunduz\PycharmProjects\PyTorch-MAML\materials\omniglot"
#
#     convert_split(os.path.join(base_dir, "train"), "train", save_dir)
#     convert_split(os.path.join(base_dir, "val"), "val", save_dir)
#     convert_split(os.path.join(base_dir, "test"), "test", save_dir)



import os
import pickle
from PIL import Image
import numpy as np
import shutil

def convert_split(split_dir, split_name, save_dir):
    """
    Omniglot klasör yapısı:
    split_dir/
        alphabet_1/
            character_1/
                img1.png
                img2.png
            character_2/
                ...
        alphabet_2/
            character_1/
                ...

    Her character klasörü 1 class kabul edilir.
    """
    all_images = []
    all_labels = []
    class_idx = 0

    alphabets = sorted(os.listdir(split_dir))
    print(f"[{split_name}] Found {len(alphabets)} alphabets.")

    for alphabet in alphabets:
        alphabet_path = os.path.join(split_dir, alphabet)
        if not os.path.isdir(alphabet_path):
            continue

        characters = sorted(os.listdir(alphabet_path))
        for character in characters:
            character_path = os.path.join(alphabet_path, character)
            if not os.path.isdir(character_path):
                continue

            imgs = os.listdir(character_path)
            for img_name in imgs:
                img_path = os.path.join(character_path, img_name)
                try:
                    img = Image.open(img_path).convert("RGB").resize((84, 84))
                    all_images.append(np.array(img))
                    all_labels.append(class_idx)
                except Exception as e:
                    print(f"⚠️ Skipping {img_path}: {e}")

            class_idx += 1

    pack = {
        "data": np.array(all_images, dtype=np.uint8),
        "labels": np.array(all_labels, dtype=np.int64),
    }

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"train_phase_{split_name}.pickle")
    with open(save_path, "wb") as f:
        pickle.dump(pack, f)

    print(f"✅ Saved {save_path}")
    print(f"   Total samples: {len(all_labels)}")
    print(f"   Total classes: {class_idx}")

if __name__ == "__main__":
    base_dir = r"C:\Users\aligunduz\PycharmProjects\PyTorch-MAML\data\omniglot"
    save_dir = r"C:\Users\aligunduz\PycharmProjects\PyTorch-MAML\materials\omniglot"

    convert_split(os.path.join(base_dir, "train"), "train", save_dir)
    convert_split(os.path.join(base_dir, "val"), "val", save_dir)
    convert_split(os.path.join(base_dir, "test"), "test", save_dir)

    # meta-val için ayrıca val.pickle bekleniyor
    src = os.path.join(save_dir, "train_phase_val.pickle")
    dst = os.path.join(save_dir, "val.pickle")
    shutil.copy(src, dst)
    print(f"✅ Copied {src} -> {dst}")