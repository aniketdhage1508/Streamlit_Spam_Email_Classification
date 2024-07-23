import streamlit as st
import os
import pickle

# Add CSS file to app
with open(os.path.join(os.path.dirname(__file__), 'styles/style.css')) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def loading_the_files():
    with open('models/logistic.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    with open('models/feature_extrac.pkl', 'rb') as f:
        feature_extraction = pickle.load(f)
        
    return loaded_model, feature_extraction


# Define pages
def home():
    st.markdown("## Spam Savvy🔎")
    st.markdown("• Web application for spam email classification and spam score calculation.")
    st.markdown("• Utilizes machine learning algorithms to analyze email content and provide accurate classifications.")
    st.markdown("• Provides users with instant feedback on the likelihood of an email being spam.")
    st.image("images/trial-removebg-preview (1).png", use_column_width=True)


    

def spam_checker():
    st.markdown("# Spam Checker")
    
    loaded_model, feature_extraction = loading_the_files()
    
    input_mail = st.text_area("Enter the email text:")
    if st.button("Check"):
        if input_mail:
            input_data_features = feature_extraction.transform([input_mail])
            prediction = loaded_model.predict(input_data_features)
            if prediction[0] == 1:
                st.success("This email is not spam.")
            else:
                st.error("This email is spam.")
        else:
            st.error("Please enter an email text.")

def spam_score():
    
    loaded_model, feature_extraction = loading_the_files()
    
    # Define the layout using Streamlit's columns
    col1, col2 = st.columns([0.1, 1])  # Adjust the column widths as needed

    # Add the image to the first column
    col1.image("images/speedometer (1).png", width=50)  # Adjust the width as needed

    # Add the text to the second column
    col2.markdown("# Spam Score")

    mail_text = st.text_area("Enter the email text:")
    if st.button("Score"):
        if mail_text:
            X = feature_extraction.transform([mail_text])
            prediction = loaded_model.predict_proba(X)
            score = prediction[0][1]
            score = 1-score
            st.success(f"The spam score is {score:.2f}.")
        else:
            st.error("Please enter an email text.")

# Add pages to app
pages = {
    "Home": home,
    "Spam Checker": spam_checker,
    "Spam Score": spam_score
}

current_page = st.sidebar.selectbox("Navbar", list(pages.keys()))
pages[current_page]()
