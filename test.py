from model import *
from tensorflow.keras.optimizers import Adam
from dataloader import *
from utils import *
from PIL import Image
import cv2
import argparse

def set_args():
  parse = argparse.ArgumentParser()
  parse.add_argument('--image_size', type = int, default = 256, help = 'image size')
  parse.add_argument('--train_image_path', type = str, default = 'C:\medical_image_segmentation\polyp\TrainDataset\images\*', help = 'train_image_path')
  parse.add_argument('--train_mask_path', type = str, default = 'C:\medical_image_segmentation\polyp\TrainDataset\masks\*', help = 'train_mask_path')
  parse.add_argument('--test_name', type = str, default = 'kvasir', help = 'test data')
  parse.add_argument('--epoch', type = int, default = 200, help = 'epoch')
  parse.add_argument('--lr', type = float, default = 1e-4, help = 'lr')
  parse.add_argument('--batch_size', type = int, default = 16, help = 'batch_size')
  parse.add_argument('--seed', type = int, default = 28, help = 'seed')
  parse.add_argument('--encoder_name', type = str, default = 'resnet50', help = 'choose type of encoder')
  parse.add_argument('--decoder_name', type = str, default = 'parallel', help = 'choose type of decoder')
  parse.add_argument('--pretrain_path', type = str, default = '', help = 'pretrain path')
  parse.add_argument('--min_lr', type = float, default = 1e-7, help = 'min learning rate')
  parse.add_argument('--patience', type = int, default = 80, help = 'the number of patience')
  parse.add_argument('--input_channels', type = int, default = 3, help = 'the number of input channels')
  return parse.parse_args()

if __name__ == '__main__':  
      args = set_args()
      val_gener, test_gener, test_set = data_test(args)
      opt = Adam(learning_rate = args.lr)                  
      model = model((args.image_size, args.image_size, args.input_channels), args)
      model.compile(optimizer=opt, loss=dice_loss, metrics=[dice,iou,precision,recall,f1])
      model.load_weights(args.pretrain_path)
      
      results = model.evaluate(test_gener, steps= len(test_set))
      dice_score = 0 
      iou_score = 0
      precision_score = 0
      recall_score = 0
      f1_score = 0
      for i in range(len(test_set)):
        img = Image.open(test_set['image_path'].iloc[i])
        img = img.resize((args.image_size, args.image_size))
        img = np.array(img) / 255.
        img = img.reshape(1,args.image_size, args.image_size,3).astype(np.float32)
        pred = model.predict(img).astype(np.float32)
        mask = cv2.resize(cv2.imread(test_set['mask_path'].iloc[i],0),(args.image_size, args.image_size))
        mask = np.array(mask) / 255.
        mask = mask>=0.5
        mask = mask.reshape(1,args.image_size, args.image_size,1)
        mask = mask.astype(np.float32)
        dice_score += dice(mask, pred)
        iou_score += iou(mask, pred)
        precision_score += precision(mask, pred)
        recall_score += recall(mask, pred)
        f1_score += f1(mask, pred)
        

      with open('result.txt', 'w') as f:
          f.write(str({'test_name':args.test_name, 'image_size':args.image_size, 'batch_size':args.batch_size, 'epoch': args.epoch, 'learning_rate': args.lr,'min_lr': args.min_lr,'patience':args.patience, 'encoder_name': args.encoder_name, 'decoder_name': args.decoder_name}))
          f.write('\n')
          f.write("Dice Score is {:.3f}".format(dice_score/len(test_set)))
          f.write('\n')
          f.write("IoU Score is {:.3f}".format(iou_score/len(test_set)))
          f.write('\n')
          f.write("Precision Score is {:.3f}".format(precision_score/len(test_set)))
          f.write('\n')
          f.write("Recall Score is {:.3f}".format(recall_score/len(test_set)))
          f.write('\n')
          f.write("F1 Score is {:.3f}".format(f1_score/len(test_set)))
          f.write('\n')
          f.write("Evaluate Result: Dice Score: {:.3f}, IoU Score: {:.3f}, Precision Score: {:.3f}, Recall Score: {:.3f}, F1 Score: {:.3f} ".format(results[1], results[2], results[3], results[4], results[5]))
          f.close()

