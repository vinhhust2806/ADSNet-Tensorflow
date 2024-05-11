import glob
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_generator(data_frame, batch_size, aug_dict,
        image_color_mode="rgb",
        mask_color_mode="grayscale",
        image_save_prefix="image",
        mask_save_prefix="mask",
        save_to_dir=None,
        target_size=256,
        seed=28):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col = "image_path",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col = "mask_path",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img,mask)

def adjust_data(img,mask):
    img = img / 255.
    mask = mask / 255.
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img, mask)

train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')

def data_train(args):
    train = pd.DataFrame()
    image_path = glob.glob(args.train_image_path)
    image_path.sort()
    mask_path = glob.glob(args.train_mask_path)
    mask_path.sort()
    train['image_path'] = image_path
    train['mask_path'] = mask_path
    
    train_gener = train_generator(train, args.batch_size,
                                train_generator_args,
                                target_size=(args.image_size,args.image_size), seed = args.seed)
    
    return train_gener, train

def data_test(args):    
    if args.test_name == 'etis':
        test = pd.DataFrame()
        image_path = glob.glob('polyp\TestDataset\ETIS-LaribPolypDB\images\*')
        image_path.sort()
        mask_path = glob.glob('polyp\TestDataset\ETIS-LaribPolypDB\masks\*')
        mask_path.sort()
        test['image_path'] = image_path
        test['mask_path'] = mask_path
    if args.test_name == 'kvasir':
        test = pd.DataFrame()
        image_path = glob.glob('polyp\TestDataset\Kvasir\images\*')
        image_path.sort()
        mask_path = glob.glob('polyp\TestDataset\Kvasir\masks\*')
        mask_path.sort()
        test['image_path'] = image_path
        test['mask_path'] = mask_path
    if args.test_name == 'clinic':
        test = pd.DataFrame()
        image_path = glob.glob('polyp\TestDataset\CVC-ClinicDB\images\*')
        image_path.sort()
        mask_path = glob.glob('polyp\TestDataset\CVC-ClinicDB\masks\*')
        mask_path.sort()
        test['image_path'] = image_path
        test['mask_path'] = mask_path
    if args.test_name == 'colon':
        test = pd.DataFrame()
        image_path = glob.glob('polyp\TestDataset\CVC-ColonDB\images\*')
        image_path.sort()
        mask_path = glob.glob('polyp\TestDataset\CVC-ColonDB\masks\*')
        mask_path.sort()
        test['image_path'] = image_path
        test['mask_path'] = mask_path
    if args.test_name == 'endoscene':
        test = pd.DataFrame()
        image_path = glob.glob(polyp\TestDataset\CVC-300\images\*')
        image_path.sort()
        mask_path = glob.glob('polyp\TestDataset\CVC-300\masks\*')
        mask_path.sort()
        test['image_path'] = image_path
        test['mask_path'] = mask_path
    
    val_gener = train_generator(test, args.batch_size,
                                dict(),
                                target_size=(args.image_size,args.image_size), seed = args.seed)
    
    test_gener = train_generator(test, 1,
                                dict(),
                                target_size=(args.image_size,args.image_size), seed = args.seed)
    
    return val_gener, test_gener, test












