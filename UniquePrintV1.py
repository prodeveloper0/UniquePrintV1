import fingerNet
import fingerData
import cGAN

import os
import time
import cv2
import numpy as np

'''
Synthesizing Fingerprint from Pattern Type Analysis Features using cGAN
World IT Congress 2019 Jeju
February 11-13, 2019, Jeju, Korea

Paper author(s): Samuel Lee and Jae-Gab Choi
{lsme8821, kor03}@ssu.ac.kr

Experimental Result
We used NIST Special Database 4 excluded worst quality fingerprint and modified size to 128x128.
And we trained the FingerNet with 150 batch size, 50000 epoch and 0.0005 learning rate both G and B parameters.
'''

# Parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
database_path = r'Resources/nist-sd4'
name = 'UniquePrintV1_FingerNet'
opt = 'demo'
checkpoint_epoch = None
batch_size = 150
epoch = 2000
D_learning_rate = 0.0005
G_learning_rate = 0.0005
max_to_keep = 999


# DNN networks
generator = fingerNet.Generator()
discriminator = fingerNet.Discriminator()


# DO IT!
def train():
    data = fingerData.FingerData(database_path, do_shuffle=True)
    cgan = cGAN.CGAN(data=data, generator=generator, discriminator=discriminator,
                     d_learning_rate=D_learning_rate, g_learning_rate=G_learning_rate, max_to_keep=max_to_keep)

    cgan.train(epoch=epoch, batch_size=batch_size, name=name)


def demo():
    cgan = cGAN.CGAN(data=fingerData.FingerData(None), generator=generator, discriminator=discriminator)
    cgan.generate(name=name, checkpoint_epoch=checkpoint_epoch)


def test():
    test_dir = os.path.join(name, 'Tests')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    cgan = cGAN.CGAN(data=fingerData.FingerData(None), generator=generator, discriminator=discriminator)
    type_index_map = ['L', 'W', 'R', 'T', 'A']
    f = open(os.path.join(test_dir, 'list.txt'), 'w')
    for i in range(100):
        samples = cgan.generate(False, name=name, checkpoint_epoch=checkpoint_epoch)
        samples = np.split(samples, 5)
        for idx_type, type in enumerate(samples):
            for idx_pattern, pattern in enumerate(type):
                filename = '{}_{}_{}.png'.format(type_index_map[idx_type], idx_pattern, i)
                filepath = os.path.join(test_dir, filename)
                cv2.imwrite(filepath, pattern)
                f.write('{},_,{}\n'.format(filename, type_index_map[idx_type]))
    f.close()


if opt == 'train':
    time_before = time.time()
    train()
    time_diff = time.time() - time_before
    print('Spent time %s' % time.strftime('%H:%M:%S', time.gmtime(time_diff)))

elif opt == 'demo':
    demo()

elif opt == 'test':
    test()
