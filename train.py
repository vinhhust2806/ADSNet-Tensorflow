from utils import *
from dataloader import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from model import *
import argparse
import random 

def set_args():
  parse = argparse.ArgumentParser()
  parse.add_argument('--image_size', type = int, default = 256, help = 'image size')
  parse.add_argument('--train_image_path', type = str, default = 'polyp\TrainDataset\images\*', help = 'train_image_path')
  parse.add_argument('--train_mask_path', type = str, default = 'polyp\TrainDataset\masks\*', help = 'train_mask_path')
  parse.add_argument('--val_name', type = str, default = 'kvasir', help = 'name of val data')
  parse.add_argument('--epoch', type = int, default = 400, help = 'epoch')
  parse.add_argument('--lr', type = float, default = 1e-4, help = 'lr')
  parse.add_argument('--batch_size', type = int, default = 16, help = 'batch_size')
  parse.add_argument('--seed', type = int, default = 28, help = 'seed')
  parse.add_argument('--encoder_name', type = str, default = 'efficientnet', help = 'choose type of encoder')
  parse.add_argument('--min_lr', type = float, default = 1e-7, help = 'min learning rate')
  parse.add_argument('--patience', type = int, default = 80, help = 'the number of patience')
  parse.add_argument('--input_channels', type = int, default = 3, help = 'the number of input channels')
  parse.add_argument('--semantic_boundary', type = float, default = 1e-5, help = 'the threshold separate early prediction')
  return parse.parse_args()

if __name__ == '__main__':
      # Initialize the seed
      args = set_args()
      np.random.seed(args.seed)
      tf.random.set_seed(args.seed)
      random.seed(args.seed)

      # Load data
      train_gener, train = data_train(args)
      val_gener, test_gener, test = data_test(args)

      # Initialize the model
      model = model((args.image_size, args.image_size, args.input_channels), args)

      # Initialize training configuration
      opt = Adam(learning_rate=args.lr)                 
      model.compile(optimizer = opt, loss = dice_loss, metrics = [dice,iou,precision,recall,f1])
      callbacks = [ModelCheckpoint(args.encoder_name+'_'+args.decoder_name+'.hdf5', monitor = 'val_dice', mode = 'max', verbose = 1, save_best_only = True)
                   ,ReduceLROnPlateau(monitor = 'val_dice', mode = 'max', factor = 0.1, patience = args.patience, min_lr = args.min_lr)]

      # Train the model
      history = model.fit(train_gener,
                    steps_per_epoch = len(train)/args.batch_size, 
                    epochs = args.epoch,
                    callbacks = callbacks,
                    validation_data = test_gener,
                    validation_steps = len(test))
