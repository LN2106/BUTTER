import streamlit as st
import pickle
import streamlit_authenticator as stauth
from pathlib import Path


st.set_page_config(
    page_title="",
    page_icon="door-arrow-right",
    layout='wide'
)

# --- USER AUTHENTICATION ---
names = ['Visitor', 'Admin']
usernames = ['user', 'admin']

# Load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                    'churn_app', 'abcdefg', cookie_expiry_days=10)


# --- LANDING PAGE ---
col1,col2 = st.columns(2)
with col2:
    name, authentication_status, username = authenticator.login('Login','main')
    st.session_state['name'] = name
    st.session_state['authentication_status'] = authentication_status
    st.session_state['username'] = username

    if authentication_status == None:
        st.info('Please Log in to continue')
        st.markdown('### **Demo credentials**')
        st.write('Username: user')
        st.write('Password: abc123')
        st.markdown('''
                    <style>
                    [data-testid="stSidebar"] {
                        visibility: hidden;
                    }
                    </style>
                    ''', unsafe_allow_html=True)
            
with col1:
    if authentication_status == None:
        st.title('This is a binary classification model built in practice within the [Azubi Africa](https://azubiafrica.org) Data Science Career Accelerator\n'
                'It has been developed with love and sweat by [Rama_Mwenda](https://github.com/Rama-Mwenda) \n\n')
        st.write('See something you like? Leave me a star on [github](https://github.com/Rama-Mwenda).⭐️😍 \n\n'
                'See something I can improve on or help you out with?\n\n'
                'Connect with me on [LinkedIn](https://www.linkedin.com/in/rama-mwenda/) and let us make magic happen :smiley:\n\n\n'
                )
        

# -- Main app section --
if authentication_status == False:
    with col1:
        st.title('This is a binary classification model built in practice within the [Azubi Africa](https://azubiafrica.org) Data Science Bootcamp\n'
                'It has been developed with love and sweat by [Rama_Mwenda](https://github.com/Rama-Mwenda) \n\n')
        st.write('See something you like? Leave me a star on [github](https://github.com/Rama-Mwenda).⭐️😍 \n\n'
                'See something I can improve on or help you out with?\n\n'
                'Connect with me on [LinkedIn](https://www.linkedin.com/in/rama-mwenda/) and let us make magic happen :smiley:\n\n\n'
                )
    with col2:
        st.error('Wrong username / password')
        st.info('Please try again with the credentials below')
        st.markdown('### **Demo credentials**')
        st.write('Username: user')
        st.write('Password: abc123')
        st.markdown('''
        <style>
        [data-testid="stSidebar"] {
            visibility: hidden;
        }
        </style>
        ''', unsafe_allow_html=True)


# --- HOME PAGE ---
if authentication_status:
    st.button(f'{st.session_state["name"]} logged in')
    st.title(f'Welcome {st.session_state["username"]} to the Customer Churn Prediction App')
    st.write('''
    <br>
    </div>
    <div style="background-color:#000;padding:10px;border-radius:10px">
    <h1 style="color:#ffd700">Predicting Customer Churn</h1>
    <p style="color:#ffffff">Predicting customer churn marks a substantial advancement in customer retention analytics.</p>
    <p style="color:#ffffff">Our app is designed to help you anticipate customer churn and take proactive measures to retain your valuable customers.</p>

    <br>
    <div style="background-color:#000;padding:10px;border-radius:10px">
    <h1 style="color:#ffd700">Unlock the Power of Data-Driven Insights</h1>
    <p style="color:#ffffff">Our app leverages advanced machine learning algorithms to analyze customer data and identify patterns that indicate a high risk.</p>
    <p style="color:#ffffff">Review and analyze past predictions, and gain a deeper understanding of your customers' behavior.</p>
    <p style="color:#ffffff">Our intuitive design ensures a seamless and efficient experience.</p>
    <p style="color:#ffffff">Simply input your customer data, select from our robust machine learning models (AdaBoost, Logistic Regression, or Random Forest),</p>
    <p style="color:#ffffff">and get instant predictions on the likelihood of customer churn.</p>
    
    <h2 style="color:#ffd700">Key Features</h2>
    <ul>
        <li style="color:#ffff00"><b>Machine Learning Models:</b> Choose from AdaBoost, Logistic Regression, or Random Forest models to find the best fit for your dataset.</li>
        <li style="color:#ffffff"><b>Streamlined Prediction Process:</b> Get quick and accurate results with our efficient prediction process.</li>
        <li style="color:#ffff00"><b>Data Retrieval:</b> Access your prediction history anytime, stored in a convenient "history.csv" file.</li>
        <li style="color:#ffffff"><b>Data Display:</b> Easily review and analyze past predictions in a clear and organized tabular format.</li>
    </ul>
   
    <h2 style="color:#ffd700">Empowering Users, Democratizing Data</h2>
    <p style="color:#ffffff">Our app goes beyond predictive analytics, democratizing data and putting the power in your hands.</p>
    <p style="color:#ffffff">Make data-driven decisions, reduce churn rates, and drive business success.</p>

    <br>

    <div style="background-color:#ffd700;padding:10px;border-radius:10px">
    <h1 style="color:#000">GET STARTED TODAY!</h1>
    <p style="color:#000">Explore our app and discover the value of accurate churn prediction.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('### ================================================ ')
    authenticator.logout('Logout','main')
    
    