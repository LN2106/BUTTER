# Churn_App
Machine Learning models embedded into a streamlit Webapp

<a name="readme-top"></a>

<div align="center">
  <h1><b>Churn Predictor</b></h1>
</div>

<!-- TABLE OF CONTENTS -->

# 🔖 Table of Contents

- [📑 Table of Contents](#-table-of-contents)
- [Churn Predictor ](#churn-cipher-)
  - [🛠 Built With ](#-built-with-)
    - [Streamlit ](#streamlit-)
  - [Key Features ](#key-features-)
  - [💻 Getting Started ](#-getting-started-)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
    - [Install](#install)
    - [Usage](#usage)
  - [☑️ Contributing ](#-contributing-)
  - [⭐️ Show your support ](#️-show-your-support-)
  - [🤝 Acknowledgments ](#-acknowledgments-)
  - [📝 License ](#-license-)
  - [🦹‍♀️ Authors ](#-authors-)

# Churn Predictor
 This is an application that implements machine learning algortihms to predict the likelyhood of a customer discountinuing usage of service for Vodafone Telco. 
 Users can interact with the Models on the prediction page, Check training data in Data Page, view visualizations on the dashboard page and see results of their input on History Page.

Features 
1. **SeniorCitizen** -- Whether a customer is a senior citizen or not
2. **Partner** -- Whether the customer has a partner or not (Yes, No)
3. **Dependents** -- Whether the customer has dependents or not (Yes, No)
4. **Tenure** -- Number of months the customer has stayed with the company
5. **Phone Service** -- Whether the customer has a phone service or not (Yes, No)
6. **MultipleLines** -- Whether the customer has multiple lines or not
7. **InternetService** -- Customer's internet service provider (DSL, Fiber Optic, No)
8. **OnlineSecurity** -- Whether the customer has online security or not (Yes, No, No Internet)
9. **OnlineBackup** -- Whether the customer has online backup or not (Yes, No, No Internet)
10. **DeviceProtection** -- Whether the customer has device protection or not (Yes, No, No internet service)
11. **TechSupport** -- Whether the customer has tech support or not (Yes, No, No internet)
12. **StreamingTV** -- Whether the customer has streaming TV or not (Yes, No, No internet service)
13. **StreamingMovies** -- Whether the customer has streaming movies or not (Yes, No, No Internet service)
14. **Contract** -- The contract term of the customer (Month-to-Month, One year, Two year)
15. **PaperlessBilling** -- Whether the customer has paperless billing or not (Yes, No)
16. **Payment Method** -- The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))
17. **MonthlyCharges** -- The amount charged to the customer monthly
18. **TotalCharges** -- The total amount charged to the customer

## 🛠 Built With <a name="built-with"></a>

### Streamlit <a name="streamlit"></a>

<details>
  <summary>GUI</summary>
  <ul>
    <li><a href="">Streamlit</a></li>
  </ul>
</details>

<details>
<summary>Database</summary>
  <ul>
    <li><a href="">Plotly</a></li>
  </ul>
</details>

<details>
<summary>Language</summary>
  <ul>
    <li><a href="">Python</a></li>
  </ul>
</details>

<details>
<summary>Model</summary>
  <ul>
    <li><a href="">Sklearn</a></li>
  </ul>
</details>

## 💨 Key Features <a name="key-features"></a>

- **A dashboard application that presents visualizations on the exploratory data and the KPIs**
- **A predicitons page to predict by specifying the model you want to use**
- **Predictions are saved for future reference and users can view the history of their prediction input values**
- **Easy model comparison on predict page to compare the performance of different models**


## 💻 Getting Started <a name="getting-started"></a>

To get a local copy up and running, follow these steps.

### Prerequisites
In order to run this project you need:
- Python

### 🏹 Setup
Clone this repository to your desired folder:
```sh
  cd my-folder
  git clone https://github.com/Rama-Mwenda/Streamlit_App.git
```
Change into the cloned repository

```sh
  cd churn_app
  
```

Create a virtual environment

```sh

python -m venv venv

```

Activate the virtual environment

```sh
    .venv/Scripts/activate
```

### Install
Here, you need to recursively install the packages in the `requirements.txt` file using the command below 

```sh
   pip install -r requirements.txt
```
### Usage
To run the project, execute the following command:

```sh
    streamlit run Home.py

```
- A webpage opens up to view the app
- Test a prediction by clicking on the predicitons page

## ☑️ Contributing <a name="contributing"></a>
Contributions, features and issues are welcome .

## ⭐️ Show your support <a name="support"></a>
If you like this project show some love with a  🌟 **STAR** 🌟

## 🤝 Acknowledgments <a name="acknowledgements"></a>
I would like to extend my appreciations to my team project and Azubi Africa for their immense support in availing all the necessary resources for this project's success.

## 📝 License <a name="license"></a> 
This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.

## 🦹‍♀️ Author <a name="authors"></a>
🕵🏽‍♀️ **Ramadhan Mwenda**

[@Rama-Mwenda](https://github.com/Rama-Mwenda) 
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/Rama-Mwenda/)