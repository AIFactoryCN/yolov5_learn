import logging
import datetime
import os
import shutil
import hashlib

from pathlib import Path
from PIL import Image, ExifTags
from logging.handlers import TimedRotatingFileHandler


for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def get_md5(data):
    return hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()

def exif_size(img):
    '''
    Returns exif-corrected PIL size
    '''
    width, height = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6 or rotation == 8:  # rotation 270  or  rotation 90
            # exchange
            width, height = height, width

    except Exception as e:
        pass
    return width, height


def mkdirs(directory):
    os.makedirs(directory, exist_ok=True)


def mkparents(path):
    parent = Path(path).parent
    if not os.path.exists(parent):
        mkdirs(parent)


def build_logger(path):
    logger = logging.getLogger("NewLogger")
    logger.setLevel(logging.INFO)
    mkparents(path)

    rf_handler = logging.handlers.TimedRotatingFileHandler(path, when='midnight', interval=1, backupCount=7, atTime=datetime.time(0, 0, 0, 0))
    formatter = logging.Formatter('[%(levelname)s][%(filename)s:%(lineno)d][%(asctime)s]: %(message)s')
    rf_handler.setFormatter(formatter)
    logger.addHandler(rf_handler)

    sh_handler = logging.StreamHandler()
    sh_handler.setFormatter(formatter)
    logger.addHandler(sh_handler)
    return logger


def build_default_logger():
    logger = logging.getLogger("DefaultLogger")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(levelname)s][%(filename)s:%(lineno)d][%(asctime)s]: %(message)s')
    sh_handler = logging.StreamHandler()
    sh_handler.setFormatter(formatter)
    logger.addHandler(sh_handler)
    return logger


def copy_code_to(src, dst):
    if len(dst) == 0 or dst == ".":
        print("invalid operate, copy to current directory")
        return

    for file in os.listdir(src):
        if file.endswith(".py"):
            source = f"{src}/{file}"
            dest = f"{dst}/{file}"
            mkparents(dest)
            shutil.copy(source, dest)


class SingleInstanceLogger:
    def __init__(self):
        self.logger = build_default_logger()

    def __getattr__(self, name):
        return getattr(self.logger, name)


def setup_single_instance_logger(path):
    global _single_instance_logger
    _single_instance_logger.logger = build_logger(path)

_single_instance_logger = SingleInstanceLogger()