import tensorflow as tf

import src.trainer as train
import src.util as util

util.clear_session()
util.config_gpu()

trainer = train.Trainer(name='test',
              net_type=train.NetType.GRD,
              batch_size=2,
              channels=3,
              epochs=1000,
              train_and_validation_path='/ArchiveEXT4/BONFI/hazelnut_2label/train',
              train_and_validation_roi_path='/ArchiveEXT4/BONFI/hazelnut_2label/ROI',
              validation_split=0.1,
              test_path='/ArchiveEXT4/BONFI/hazelnut_2label/test',
              mask_path='/ArchiveEXT4/BONFI/hazelnut_2label/MASK',
              patch_size=[64, 64],
              patches_row=1,
              patches_col=1,
              stride=[64, 64],
              padding='VALID',
              mask_suffix='mask')
trainer.train()

trainer.show_first_batch_images_and_masks(True, False)
trainer.show_first_batch_images_and_masks(True, True)
trainer.show_first_batch_images_and_masks(False, False)
trainer.show_first_batch_images_and_masks(False, True)
