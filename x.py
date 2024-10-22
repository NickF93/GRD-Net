import tensorflow as tf

import src.trainer as train
import src.util as util

util.clear_session()
util.config_gpu()

trainer = train.Trainer(name='test',
              net_type=train.NetType.GRD,
              batch_size=32,
              latent_size=512,
              channels=3,
              epochs=1000,
              train_and_validation_path='/mnt/hdd/mvtec_anomaly_detection/hazelnut_2label/train',
              train_and_validation_roi_path='/mnt/hdd/mvtec_anomaly_detection/hazelnut_2label/ROI',
              validation_split=0.1,
              test_path='/mnt/hdd/mvtec_anomaly_detection/hazelnut_2label/test',
              mask_path='/mnt/hdd/mvtec_anomaly_detection/hazelnut_2label/MASK',
              patch_size=[128, 128],
              patches_row=1,
              patches_col=1,
              stride=[128, 128],
              padding='VALID',
              mask_suffix='mask',
              initial_learning_rate=1e-4,
              first_decay_steps=1000,
              t_mul=2.0,
              m_mul=(1.0 / 2.0),
              alpha=(1.0 / 25.0))
trainer.train()

trainer.show_first_batch_images_and_masks(True, False)
trainer.show_first_batch_images_and_masks(True, True)
trainer.show_first_batch_images_and_masks(False, False)
trainer.show_first_batch_images_and_masks(False, True)
