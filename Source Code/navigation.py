import streamlit as st

import base64
# import numpy as np
# import matplotlib.pyplot as plt 
# from tkinter.filedialog import askopenfilename
# import cv2
# import streamlit as st
# from PIL import Image
# import matplotlib.image as mpimg
# import cv2
# from tensorflow.keras.layers import Dense, Conv2D
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.models import Sequential
# import streamlit as st
# import base64
# from keras.utils import to_categorical

# ================ Background image ===

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


def navigation():
    try:
        path = st.experimental_get_query_params()['p'][0]
    except Exception as e:
        st.error('Please use the main app.')
        return None
    return path


if navigation() == "home":
    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Skin disease prediction using neural networks with remedy recommendation"}</h1>', unsafe_allow_html=True)

    # st.write("Alzheimer's is a type of dementia that causes problems with memory, thinking and behaviour. Symptoms usually develop slowly and get worse over time, becoming severe enough to interfere with daily tasks. Dementia is not a specific disease. It’s an overall term that describes a group of symptoms associated with a decline in memory or other thinking skills severe enough to reduce a person’s ability to perform everyday activities. Alzheimer’s disease accounts for 60 to 80 percent of cases. Vascular dementia, which occurs after a stroke, is the second most common dementia type. But there are many other conditions that can cause symptoms of dementia, including some that are reversible, such as thyroid problems and vitamin deficiencies. Dementia is a general term for loss of memory and other mental abilities severe enough to interfere with daily life. It is caused by physical changes in the brain. Alzheimer’s is the most common type of dementia, but there are many kinds. The input data is taken from the dataset repository. In our process, we are take the Alzheimer’s disease dataset as input. The system is developed the deep learning algorithm . The results shows that the performances metrics such as accuracy and predict the disease as mild, moderate, very moderate and dementia.")

    # st.title('Home')
        # ================== REMEDIES ===========================
    
  

elif navigation()=='login':
    st.title("Welcome Login Page !!!")

    import pandas as pd
    
    # df = pd.read_csv('login_record.csv')
    
    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False
    
    col1, col2 = st.columns(2)
    
    
        
    with col1:
    
        UR1 = st.text_input("Login User Name",key="username")
        psslog = st.text_input("Password",key="password",type="password")
        # tokenn=st.text_input("Enter Access Key",key="Access")
        agree = st.checkbox('LOGIN')
        
        if agree:
            try:
                
                df = pd.read_csv(UR1+'.csv')
                U_P1 = df['User'][0]
                U_P2 = df['Password'][0]
                if str(UR1) == str(U_P1) and str(psslog) == str(U_P2):
                    st.success('Successfully Login !!!')    

            
                    import pandas as pd
                     
                    def hyperlink(url):
                         return f'<a target="_blank" href="{url}">{url}</a>'
                     
                    dff = pd.DataFrame(columns=['page'])
                    dff['page'] = ['Question']
                    dff['page'] = dff['page'].apply(hyperlink)
                    dff = dff.to_html(escape=False)
     
                    st.write(dff, unsafe_allow_html=True)   
    
                else:
                    st.write('Login Failed!!!')
            except:
                st.write('Login Failed!!!')                 
    with col2:
        UR = st.text_input("Register User Name",key="username1")
        pss1 = st.text_input("First Password",key="password1",type="password")
        pss2 = st.text_input("Confirm Password",key="password2",type="password")
        # temp_user=[]
            
        # temp_user.append(UR)
        
        if pss1 == pss2 and len(str(pss1)) > 2:
            import pandas as pd
            
      
            import csv 
            
            # field names 
            fields = ['User', 'Password'] 
            

            
            # st.text(temp_user)
            old_row = [[UR,pss1]]
            
            # writing to csv file 
            with open(UR+'.csv', 'w') as csvfile: 
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile) 
                    
                # writing the fields 
                csvwriter.writerow(fields) 
                    
                # writing the data rows 
                csvwriter.writerows(old_row)
            st.success('Successfully Registered !!!')
        else:
            
            st.write('Registeration Failed !!!')     

    
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by Yuvaraj</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)      
