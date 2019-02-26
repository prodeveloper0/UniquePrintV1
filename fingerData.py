import numpy as np
import cv2
import os
import random


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]


class FingerData(object):
    def __init__(self, path, index='list.txt', except_index='except.txt', do_shuffle=True):
        # Database info
        self.name = 'NIST Special Database 4'
        self.z_dim = 128
        self.y_dim = 5
        self.size = 128
        self.channel = 1

        # Fingerprint data
        self.image_list = list()
        self.image_type_list = list()

        # For batch
        self.current_index = 0

        if path == '' or path is None:
            return

        # Read database index
        list_filename = os.path.join(path, index)
        database_line = list()
        with open(list_filename) as f:
            while True:
                line = f.readline()
                if line.strip() == '':
                    break
                database_line.append(line.strip())

        # Read except index
        except_list_filename = os.path.join(path, except_index)
        except_database_list = list()
        with open(except_list_filename) as f:
            while True:
                line = f.readline()
                if line.strip() == '':
                    break
                except_database_list.append(line.strip())

        # Shuffle database index
        if do_shuffle:
            random.shuffle(database_line)

        # Read images and labels
        image_type_dict = {'L': np.array([1, 0, 0, 0, 0], dtype=np.float32),
                           'W': np.array([0, 1, 0, 0, 0], dtype=np.float32),
                           'R': np.array([0, 0, 1, 0, 0], dtype=np.float32),
                           'T': np.array([0, 0, 0, 1, 0], dtype=np.float32),
                           'A': np.array([0, 0, 0, 0, 1], dtype=np.float32)}

        for line in database_line:
            split_line = line.split(",")

            if split_line[0] in except_database_list:
                continue

            img = cv2.imread(os.path.join(path, split_line[0]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Normalize image
            # img = cv2.equalizeHist(img)
            img = np.array(img).astype(dtype=np.float32) / 255.

            self.image_list.append(img.reshape(img.shape + (1,)))
            self.image_type_list.append(image_type_dict[split_line[2]])

    def __call__(self, batch_size):
        return list(chunks(self.image_list, batch_size)), list(chunks(self.image_type_list, batch_size))
