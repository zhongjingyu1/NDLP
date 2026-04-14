import os.path as osp
import numpy as np
import mmcv
from PIL import Image

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class NUSWideDataset(CustomDataset):
    """NUS-WIDE multi-label dataset using plain text annotation lists.

    Expected ann_file format (one sample per line):
        relative/path/to/image.jpg cls_idx_1 cls_idx_2 ...

    Example:
        images/158175_2489836051_e0876ff547_m.jpg 3 8 42

    Notes:
    - `img_prefix` should point to the NUS-WIDE root if your ann lines start with
      `images/...`, e.g. img_prefix='F:/mnt/SSD/det/nuswide/'.
    - `ann_file` can be the generated `nw_lt_train.txt` / `nw_lt_test.txt`.
    - `class_split` can use the generated `class_split.pkl`.
    """

    CLASSES = (
        'airport', 'animal', 'beach', 'bear', 'birds', 'boats', 'book', 'bridge',
        'buildings', 'cars', 'castle', 'cat', 'cityscape', 'clouds', 'computer',
        'coral', 'cow', 'dancing', 'dog', 'earthquake', 'elk', 'fire', 'fish',
        'flags', 'flowers', 'food', 'fox', 'frost', 'garden', 'glacier', 'grass',
        'harbor', 'horses', 'house', 'lake', 'leaf', 'map', 'military', 'moon',
        'mountain', 'nighttime', 'ocean', 'person', 'plane', 'plants', 'police',
        'protest', 'railroad', 'rainbow', 'reflection', 'road', 'rocks', 'running',
        'sand', 'sign', 'sky', 'snow', 'soccer', 'sports', 'statue', 'street',
        'sun', 'sunset', 'surf', 'swimmers', 'tattoo', 'temple', 'tiger', 'tower',
        'town', 'toy', 'train', 'tree', 'valley', 'vehicle', 'water', 'waterfall',
        'wedding', 'whales', 'window', 'zebra'
    )

    def __init__(self, **kwargs):
        super(NUSWideDataset, self).__init__(**kwargs)
        self.categories = self.CLASSES
        self.index_dic = self.get_index_dic()

    def load_annotations(self, ann_file, LT_ann_file=None):
        img_infos = []
        lines = mmcv.list_from_file(ann_file)

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 1:
                continue

            rel_path = parts[0]
            gt_labels = np.zeros((len(self.CLASSES),), dtype=np.int64)

            if len(parts) > 1:
                for p in parts[1:]:
                    cls_idx = int(p)
                    if cls_idx < 0 or cls_idx >= len(self.CLASSES):
                        raise ValueError(
                            f"Invalid class index {cls_idx} in {ann_file} line {idx + 1}"
                        )
                    gt_labels[cls_idx] = 1

            full_path = osp.join(self.img_prefix, rel_path)
            if not osp.exists(full_path):
                raise FileNotFoundError(f'NUSWideDataset cannot find image: {full_path}')

            # custom.py -> prepare_train_img needs height and width
            with Image.open(full_path) as img:
                width, height = img.size

            img_infos.append(
                dict(
                    id=idx,
                    filename=rel_path,
                    width=width,
                    height=height,
                    ann=dict(labels=gt_labels)
                )
            )
        return img_infos

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']
