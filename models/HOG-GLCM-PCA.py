def SVM(C=1, kernel='linear', gamma='scale', distance=1, angle=0, visualize=False):
    
    hog_train_test_txt = ('D:\\ClarentiCelsia\\Personal\\SKRIPSI\\Proposal\\HOG_GLCM[%s][%s].txt' %(distance, angle))
  
    img_names, labels, hog_features = load_txt(hog_train_test_txt)
   
    X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=0)
    
    # Apply Scalling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("X_train before PCA: ",X_train, "\n")
    print("X_train before PCA: ",X_train.shape, "\n")
    
    # Reduce dimension into 2D
      # Variance value : how much percentage of the obtained information 
    pca = PCA(n_components = 75)
    pca.fit(X_train)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    
    # Standardizing
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)
    
     # PCA Results
    print(pca.explained_variance_ratio_)
    print("\nFeatures after reducing by PCA: \n", X_train.shape)
    print("y_train: ",y_train.shape, "\n")

    # Build model svm
    svm = SVC(C=C, kernel=kernel, gamma=gamma, tol=1e-3, probability=True)
    # fit to train_model
    svm.fit(X_train, y_train)
    
    print('Accuracy of SVM classifier on training set: {:.5f}'
     .format(svm.score(X_train, y_train)))
    print('Accuracy of SVM classifier on testing set: {:.5f}'
     .format(svm.score(X_test, y_test)))
   
    #prediction
      # predict which class(y) of new dataset(X ) 
    y_pred = svm.predict(X_test)
    print("predict svm: \n", y_pred, "\n")
  
    # Accuracy of prediction
    print("Accuracy SVM: " + str(accuracy_score(y_pred, y_test)))
   
    confusionmatrixSVM = confusion_matrix(y_test, y_pred)
    print("Matrix SVM: \n", confusionmatrixSVM)
    
    classificationreport = classification_report(y_test, y_pred)
    print("SVM Report: \n", classificationreport)
   
    # using cross val
    test_scores = cross_val_score(svm, X_test, y_test, scoring='accuracy', cv=7)
    print("Accuracy of testing set by using cross val : ",test_scores)
    print("Mean of test scores: ", test_scores.mean())
    
    train_scores = cross_val_score(svm, X_train, y_train, scoring='accuracy', cv=7)
    print("Accuracy of training set by using cross val : ",train_scores)
    print("Mean of test scores: ", train_scores.mean(), "\n")
    
    """
    predict_proba_value = svm.predict_proba(X_test) 
    y_pred_valid = svm.predict(X_test)  
    
    print("what value of prediction probability so system can determine which class of them: ", predict_proba_value)
    print("what system predicts toward the validating set of svm: ", y_pred_valid)
    """
  
    print ("finished.")
    
    if visualize:
        plt.figure()
        for i, j in enumerate(np.unique(y_train)):
            plt.scatter(X_train[y_train == j, 0], X_train[y_train == j, 1],
                        s=20, c = ListedColormap(('red', 'green', 'blue', 'yellow', 'gray'))(i), label = j)
        
        plt.title("HOG")
        plt.legend()
        plt.show()
        
        visualized(folderpath)
        
def visualized(path):
        orientations = 9
        pixels_per_cell = (8, 8)
        cells_per_block = (2, 2)
        label = 0
    
        for folder in os.listdir(path):
            label = label + 1
            file_path = "".join((path,"/",folder)) #'D:\\ClarentiCelsia\\Personal\\SKRIPSI\\Proposal\\Resize_Leaf/leafA'
        
            for file in os.listdir(file_path):
                image_path = "".join((file_path, "/", file)) #'D:\\ClarentiCelsia\\Personal\\SKRIPSI\\Proposal\\Resize_Leaf/leafA/leaf001.jpg'

                # Read Image
                img = cv2.imread(image_path) # open the file
           
                #resize_img = resize(img, (128,64))
                grays = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Resize Img
                dim = (64, 128)
                resized = cv2.resize(grays, dim, interpolation = cv2.INTER_AREA)
            
                j=[]
                for i,j in enumerate(np.unique(label)):
                    #(0,1),(0,1)..(0,2)
                    j=j
    
       
            for ii, jj in enumerate(np.unique(j)):
                #1,2,3,4,5
                print("jj: ", jj)  
                fd, hog_img = hog(resized, orientations, pixels_per_cell, cells_per_block, block_norm='L2' ,feature_vector=True, visualize=True)# fd= feature descriptor
       
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True) 
                ax1.imshow(resized, cmap=plt.cm.gray) 
                ax1.set_title('Input image') 

                # Rescale histogram for better display 
                hog_image_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10)) 

                ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray) 
                ax2.set_title('Histogram of Oriented Gradients')
            
                plt.show()
   
        print("Extract successfully!")
            


if __name__ == "__main__":
    
   
    """
    GET_HOG_GLCM(img_dir = folderpath, 
            dim=(64,128),
            bins=9, 
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            norm='L2',
            distance=1,
            angle=0
            )
    """
    #SVM(C=1, kernel='linear', gamma=10, distance=1, angle=0, visualize=False)
    Naive(distance=1, angle=135)
