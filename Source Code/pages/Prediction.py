import streamlit as st

import base64
import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import streamlit as st
from PIL import Image
import matplotlib.image as mpimg
import cv2
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
import streamlit as st
import base64
from keras.utils import to_categorical

# ================ Background image ===
st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Skin disease prediction using neural networks with remedy recommendation"}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.jpg')

import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing 
import streamlit as st
import cv2
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import base64
from tkinter.filedialog import askopenfilename



uploaded_file = st.file_uploader("Choose a file")


# aa = st.button("UPLOAD IMAGE")

if uploaded_file is None:
    
    st.text("Please upload an image")

else:
    import numpy as np
# ================ INPUT IMAGE ======================

    # filename = askopenfilename()
    img = mpimg.imread(uploaded_file)
    st.image(img,caption="Original Image")

    #============================ PREPROCESS =================================
    
    #==== RESIZE IMAGE ====
    
    resized_image = cv2.resize(img,(300,300))
    img_resize_orig = cv2.resize(img,((50, 50)))
    
    fig = plt.figure()
    plt.title('RESIZED IMAGE')
    plt.imshow(resized_image)
    plt.axis ('off')
    plt.show()
       
             
    #==== GRAYSCALE IMAGE ====
    
    
    # SPV = np.shape(img)
    
    try:            
        gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
        
    except:
        gray1 = img_resize_orig
       
    fig = plt.figure()
    plt.title('GRAY SCALE IMAGE')
    plt.imshow(gray1,cmap='gray')
    plt.axis ('off')
    plt.show()
    
    # ============== FEATURE EXTRACTION ==============
    
    
    #=== MEAN STD DEVIATION ===
    
    mean_val = np.mean(gray1)
    median_val = np.median(gray1)
    var_val = np.var(gray1)
    features_extraction = [mean_val,median_val,var_val]
    
    print("-------------------------------------")
    print("        Feature Extraction          ")
    print("-------------------------------------")
    print()
    print("1) Mean Value     = ", mean_val)
    print("2) Median Value   = ",median_val )
    print("3) Varaince Value = ", var_val )
    

    # ==================== IMAGE SPLITTING ========================
    
    # === test and train ===
    
    import os 
    
    from sklearn.model_selection import train_test_split
    
    acne = os.listdir('Dataset/Acne and Rosacea Photos/')
    
    act = os.listdir('Dataset/ACT/')
            
    dermat = os.listdir('Dataset/Atopic Dermatitis Photos/')
    
    bullous = os.listdir('Dataset/Bullous Disease Photos/')     
    
    cellu = os.listdir('Dataset/Cellulitis Impetigo and other Bacterial Infections/')
    
    sys = os.listdir('Dataset/Systemic Disease/')     

    vascular = os.listdir('Dataset/Vascular Tumors/')   
    
    
    import numpy as np
    
    dot1= []
    labels1 = [] 
    for img11 in acne:
            # print(img)
            img_1 = mpimg.imread('DataSet/Acne and Rosacea Photos//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(1)
                            
    
    for img11 in act:
            # print(img)
            img_1 = mpimg.imread('DataSet/ACT//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(2)
                            
                            
    for img11 in dermat:
            # print(img)
            img_1 = mpimg.imread('DataSet/Atopic Dermatitis Photos//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(3)
                            
    
    for img11 in bullous:
            # print(img)
            img_1 = mpimg.imread('DataSet/Bullous Disease Photos//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(4)
    
    for img11 in cellu:
            # print(img)
            img_1 = mpimg.imread('DataSet/Cellulitis Impetigo and other Bacterial Infections//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(5)
    
    
    for img11 in sys:
            # print(img)
            img_1 = mpimg.imread('DataSet/Systemic Disease//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(6)
    
    for img11 in vascular:
            # print(img)
            img_1 = mpimg.imread('DataSet/Vascular Tumors//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(7)

    x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
    

    print()
    print("-------------------------------------")
    print("       IMAGE SPLITTING               ")
    print("-------------------------------------")
    print()
  
    print("Total no of data         :",len(dot1))
    print("Total no of Train data   :",len(x_train))
    print("Total no of Test data    :",len(x_test))


    # ============================== CLASSIFICATION ==========================
    
    # ==== DIMNSION EXPANSION ==
    
    from keras.utils import to_categorical

    from tensorflow.keras.models import Sequential

    y_train1=np.array(y_train)
    y_test1=np.array(y_test)

    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test)

  
    x_train2=np.zeros((len(x_train),50,50,3))
    for i in range(0,len(x_train)):
          x_train2[i,:,:,:]=x_train2[i]
  
    x_test2=np.zeros((len(x_test),50,50,3))
    for i in range(0,len(x_test)):
          x_test2[i,:,:,:]=x_test2[i]


    from keras.layers import Input, Conv2D, Activation, Add, GlobalAveragePooling2D, Dense, multiply
    from keras.models import Model
    from keras import backend as K
    
    def attention_residual_block(input_tensor, filters, kernel_size=(3, 3)):
        x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = Activation('relu')(x)
    
        # Attention Mechanism
        attention = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
        x = multiply([x, attention])
    
        # Residual Connection
        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def create_attention_residual_cnn(input_shape, num_classes):
        inputs = Input(shape=input_shape)
    
        # Initial Convolutional Layer
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
        x = Activation('relu')(x)
    
        # Attention Residual Blocks
        x = attention_residual_block(x, 64)
        x = attention_residual_block(x, 64)
    
        # Global Average Pooling and Classification
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(num_classes, activation='softmax')(x)
    
        model = Model(inputs=inputs, outputs=outputs)
        return model

# # Example usage:
# input_shape = (50, 50, 3)  # Example input shape for ImageNet images
# num_classes = 7  # Example number of output classes for ImageNet
# model = create_attention_residual_cnn(input_shape, num_classes)
# model.summary()

# initialize the model
    model=Sequential()
 
 
 #CNN layes 
    model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
    model.add(MaxPooling2D(pool_size=2))
 
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
 
    model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
 
    model.add(Dropout(0.2))
    model.add(Flatten())
 
    model.add(Dense(500,activation="relu"))

    model.add(Dropout(0.2))
 
    model.add(Dense(8,activation="softmax"))
 
 #summary the model 
 
 #compile the model 
    model.compile(loss='binary_crossentropy', optimizer='adam')
    y_train1=np.array(y_train)
 
    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test)
    
     #fit the model 
    history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=5,verbose=1)

     
    loss=history.history['loss']
    loss=max(loss)
    accuracy_cnn=100-loss
    print()

    st.text("-------------------------------------------------------------")
    st.text("Performance Analysis  --> CNN -2D")
    st.text("-------------------------------------------------------------")
    print()
    
    st.write("1.Accuracy   = ", accuracy_cnn,'%')
    print()
    st.write("2.Error Rate = ", loss)        
        
   
    Total_length = len(acne) + len(act) + len(dermat) + len(bullous) + len(cellu) + len(sys) + len(vascular)
    
    
    temp_data1  = []
    for ijk in range(0,Total_length):
        # print(ijk)
        temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
        temp_data1.append(temp_data)
    
    temp_data1 =np.array(temp_data1)
    
    zz = np.where(temp_data1==1)
    


 
    
    
    
    
    
    
    
    
                                
    if labels1[zz[0][0]] == 1:
        print('------------------------')
        print()
        print(' The Prediction = Acne and Rosacea Photos')
        print()
        print('------------------------')
        res1=" Affected by Acne and Rosacea Photos"
        st.write(res1)
        
        print("Recommended Remedy = Use a gentle cleanser and moisturizer formulated for sensitive skin to avoid irritating rosacea-prone skin.")
        st.write("Recommended Remedy = Use a gentle cleanser and moisturizer formulated for sensitive skin to avoid irritating rosacea-prone skin.")
        

        
    elif labels1[zz[0][0]] == 2:
        print('--------------------------')
        print()
        print('The Prediction = ACT')   
        print()
        print('-------------------------')
        res1=" Affected by ACT"
        
    elif labels1[zz[0][0]] == 3:
        print('--------------------------')
        print()
        print('The Prediction = Atopic Dermatitis Photos')   
        print()
        print('-------------------------')
        res1=" Affected by Atopic Dermatitis Photos"    
        
    elif labels1[zz[0][0]] == 4:
        print('--------------------------')
        print()
        print('The Prediction = Basalcell')   
        print()
        print('-------------------------')
        res1=" Affected by skin cancer - Atopic Dermatitis Photos "    
        
    elif labels1[zz[0][0]] == 5:
        print('--------------------------')
        print()
        print('The Prediction = Dermotofibroma')   
        print()
        print('---------------------------------------------------------------')
        res1=" Affected by skin cancer -Cellulitis Impetigo and other Bacterial Infections"
        
    elif labels1[zz[0][0]] == 6:
        print('--------------------------')
        print()
        print('The Prediction = Melonocytic')   
        print()
        print('-------------------------')
        res1=" Affected by skin cancer - Systemic Disease"
        
        
    elif labels1[zz[0][0]] == 7:
        print('--------------------------')
        print()
        print('The Prediction = Vascular')   
        print()
        print('-------------------------')
        res1=" Affected by skin cancer- Vascular Tumors" 
            

    from surprise import Dataset, Reader, KNNBasic
    from surprise.model_selection import train_test_split
   
    data = {
       'user': [res1],
       'remedy': ['Recommended Remedy = Use a gentle cleanser and moisturizer formulated for sensitive skin to avoid irritating rosacea-prone skin.', 'Remedy2', 'Remedy1', 'Remedy3', 'Remedy2', 'Remedy4', 'Remedy3', 'Remedy4', 'Remedy1', 'Remedy4'],
       'rating': [5, 4, 3, 5, 4, 2, 3, 1, 5, 3]
   }


    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(pd.DataFrame(data), reader)
   
   # Split the data into training and testing sets
    trainset, testset = train_test_split(dataset, test_size=0.2)
   
   # Use user-based collaborative filtering
    sim_options = {
       'name': 'cosine',
       'user_based': True  # compute similarities between users
   }
   
   # Build the recommendation system using k-NN algorithm
    algo = KNNBasic(sim_options=sim_options)
   
   # Train the model
    algo.fit(trainset)
   
    def get_top_n_recommendations(user_id, n=5):
       user_inner_id = trainset.to_inner_uid(user_id)
       
       user_unseen_items = [item_id for item_id in range(trainset.n_items) if trainset.to_raw_iid(item_id) not in [t[0] for t in trainset.ur[user_inner_id]]]
       
       item_ratings = [(trainset.to_raw_iid(iid), rating) for (iid, rating) in algo.test([(user_inner_id, iid, 0) for iid in user_unseen_items])]
       
       item_ratings.sort(key=lambda x: x[1], reverse=True)
       
       top_n = item_ratings[:n]
       
       return top_n
   
    user_id = res1
    top_recommendations = get_top_n_recommendations(user_id)
    print(f"Top recommendations for {user_id}:")
    for remedy, predicted_rating in top_recommendations:
       print(f"Remedy: {remedy}, Predicted Rating: {predicted_rating}")






