# Download dataset cats and dogs
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O /tmp/cats_and_dogs_filtered.zip
  
local_zip = '/tmp/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

base_dir = '/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train') # /tmp/cats_and_dogs_filtered/train

def resize_image(base_path, dim=(150, 150)):
  new_train_dir = os.path.join(base_path, 'train_150x150')
  
  train_datas = os.listdir(train_dir) # return list of folder train

  try:
    for folder in train_datas:  # >>> ['dogs', 'cats'] 
      os.makedirs((new_train_dir + "//" + folder))
      for items in os.listdir("".join((train_dir, "/", folder))):
        img_path = (os.path.join(train_dir, folder) + "/" + items)
        imgp = cv2.imread(img_path)
        img = cv2.cvtColor(imgp, cv2.COLOR_BGR2GRAY) # convert to grayscale
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        # Visualize resized image
        # io.imshow(img)
        # plt.show()

        # Save to new folder
        cv2.imwrite((new_train_dir + "//" + folder + "//" + items), img)

  except FileExistsError:
    print('directory is already exist')

resize_image(base_path=base_dir)
