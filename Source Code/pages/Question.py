import streamlit as st
import base64


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


st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Symptoms"}</h1>', unsafe_allow_html=True)
    
a1 = st.text_input("Is there any skin rashes or itching ?")

a2 = st.text_input("Is there any skin discoloration ?")    

a3 = st.text_input("Is there any pain in skin ?")   

a4 = st.text_input("Is there any dryness?")   

# aa =[a1,a2,a3,a4]

# aa = max(aa)

aa = st.button("Submit")

if aa:
    
    import pandas as pd
     
    def hyperlink(url):
         return f'<a target="_blank" href="{url}">{url}</a>'
     
    dff = pd.DataFrame(columns=['page'])
    dff['page'] = ['Prediction']
    dff['page'] = dff['page'].apply(hyperlink)
    dff = dff.to_html(escape=False)
 
    st.write(dff, unsafe_allow_html=True)   

else:
    st.text("Enter sysmptoms")

