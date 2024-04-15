from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from tqdm.notebook import tqdm

class TQDMNotebookCallback(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.progress_bar = tqdm(total=self.epochs)
        self.progress_bar.set_description('Training Progress')

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.update(1)

def load_data():
    train_path = Path("data/training_images")
    test_path = Path("data/testing_images")
    train = pd.read_csv("data/train_solution_bounding_boxes.csv")
    train[['xmin', 'ymin', 'xmax', 'ymax']] = train[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
    train.drop_duplicates(subset='image', inplace=True, ignore_index=True)
    return test_path, train_path, train

def data_generator(df, batch_size, path, augmentation=None):
    test_path, train_path, train = load_data()
    if augmentation is not None:
        aug = augmentation.flow(np.zeros((1, 380, 676, 3)), batch_size=batch_size, shuffle=False)  # Use passed augmentation
    else:
        aug = None  # Handle case where no augmentation is provided
    while True:        
        images = np.zeros((batch_size, 380, 676, 3))
        bounding_box_coords = np.zeros((batch_size, 4))
        
        for i in range(batch_size):
            rand_index = np.random.randint(0, train.shape[0])
            row = df.loc[rand_index, :]
            image_path = str(train_path/row.image)
            print(f"Attempting to load image from path: {image_path}")  # Debugging line
            image = cv2.imread(image_path) / 255.
            if aug is not None:
                image = aug.random_transform(image)  # Apply passed augmentation
            images[i] = image
            bounding_box_coords[i] = np.array([row.xmin, row.ymin, row.xmax, row.ymax])
                
        yield {'image': images}, {'coords': bounding_box_coords}

def load_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)  # Resize as per your model's requirement
    img = img / 255.0  # Normalize if your model expects normalized inputs
    return img

def display_image(img, bbox_coords=[], pred_coords=[], norm=False):
    # if the image has been normalized, scale it up
    if norm:
        img *= 255.
        img = img.astype(np.uint8)
    
    # Draw the bounding boxes
    if len(bbox_coords) == 4:
        xmin, ymin, xmax, ymax = bbox_coords
        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
        
    if len(pred_coords) == 4:
        xmin, ymin, xmax, ymax = pred_coords
        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 3)
        
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
def display_image_from_file(name, bbox_coords=[]):
    test_path, train_path, train = load_data()
    img = cv2.imread(str(train_path/name))
    display_image(img, bbox_coords=bbox_coords)
    
def display_from_dataframe(row):
        display_image_from_file(row['image'], bbox_coords=(row.xmin, row.ymin, row.xmax, row.ymax))
    

def display_grid(df, n_items=10):
    plt.figure(figsize=(20, 10))
    
    # Assuming you want a grid of 5x2 (5 columns, 2 rows)
    cols = 5
    rows = n_items // cols
    
    rand_indices = [np.random.randint(0, df.shape[0]) for _ in range(n_items)]
    
    for i, index in enumerate(rand_indices):
        plt.subplot(rows, cols, i + 1)
        row = df.loc[index, :]
        test_path, train_path, train = load_data()  # Load paths correctly
        img = cv2.imread(str(train_path/row['image']))
        if img is None:
            print(f"Failed to load image: {train_path/row['image']}")
            continue  # Skip this iteration if the image failed to load
        bbox_coords = (row.xmin, row.ymin, row.xmax, row.ymax)
        display_image(img, bbox_coords=bbox_coords)
        rand_indices = [np.random.randint(0, df.shape[0]) for _ in range(n_items)]
        
        for pos, index in enumerate(rand_indices):
            plt.subplot(2, n_items // 2, pos + 1)
            display_from_dataframe(df.loc[index, :])


# # Example use of display_grid function with a sample dataframe
# import pandas as pd

# # Sample dataframe
# data = {
#     'image': ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg',
#               'image6.jpg', 'image7.jpg', 'image8.jpg', 'image9.jpg', 'image10.jpg'],
#     'xmin': [30, 50, 20, 45, 55, 65, 35, 25, 60, 40],
#     'ymin': [60, 70, 50, 65, 75, 85, 55, 45, 80, 60],
#     'xmax': [130, 150, 120, 145, 155, 165, 135, 125, 160, 140],
#     'ymax': [160, 170, 150, 165, 175, 185, 155, 145, 180, 160]
# }

# df = pd.DataFrame(data)

# # Display grid of images with bounding boxes
# display_grid(df, n_items=10)
