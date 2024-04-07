from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    train_path = Path("../data/training_images")
    test_path = Path("../data/testing_images")
    train = pd.read_csv("../data/train_solution_bounding_boxes (1).csv")
    train[['xmin', 'ymin', 'xmax', 'ymax']] = train[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
    train.drop_duplicates(subset='image', inplace=True, ignore_index=True)
    return test_path, train_path, train

from augmentation import simple_augmentation  # Add this import

def data_generator(df=train, batch_size=16, path=train_path):
    test_path, train_path, train = load_data()
    aug = simple_augmentation().flow(np.zeros((1, 380, 676, 3)), batch_size=batch_size, shuffle=False)  # Initialize augmentation
    while True:        
        images = np.zeros((batch_size, 380, 676, 3))
        bounding_box_coords = np.zeros((batch_size, 4))
        
        for i in range(batch_size):
            rand_index = np.random.randint(0, train.shape[0])
            row = df.loc[rand_index, :]
            image = cv2.imread(str(train_path/row.image)) / 255.
            image = aug.random_transform(image)  # Apply augmentation
            images[i] = image
            bounding_box_coords[i] = np.array([row.xmin, row.ymin, row.xmax, row.ymax])
                
        yield {'image': images}, {'coords': bounding_box_coords}

# Test the generator
# example, label = next(data_generator(batch_size=1))
# img = example['image'][0]
# bbox_coords = label['coords'][0]

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
        
        # get 10 random entries and plot them in a 2x5 grid
        rand_indices = [np.random.randint(0, df.shape[0]) for _ in range(n_items)]
        
        for pos, index in enumerate(rand_indices):
            plt.subplot(2, n_items // 2, pos + 1)
            display_from_dataframe(df.loc[index, :])

# display_image_from_file("vid_4_10520.jpg")

# display_image(img, bbox_coords=bbox_coords, norm=True)
# display_grid(train)
