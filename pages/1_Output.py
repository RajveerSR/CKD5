import pandas as pd
import streamlit as st
import plotly.express as px  

st.title("Patient Dashboard")
st.sidebar.title("Queries or Issues")
st.sidebar.info(
    """
    For any issues with app usage, please contact: n1009755@my.ntu.ac.uk
    """
)

a, b, c = st.sidebar.columns([0.2, 1, 0.2])
with b:
    st.markdown(
        """
        <div align=center>
        <small>
        Helpful links: https://www.kidney.org/kidney-topics/chronic-kidney-disease-ckd
        </small>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.expander("ℹ️ General Information", expanded=False).markdown(
    """
    ### Machine Learining Model
    This Web App uses Logisitc Regression (LR) to predict the risk of Chronic Kidney Disease (CKD) based on various risk factors. The model is trained on the Chronic Kidney Disease dataset from the UC Irvine's Machine Learning Repository. Within the Data preprocessing stage, the dataset was cleaned and transformed to ensure the model could be trained effectively any missing values were fixed by using Verstack a Machine Learning repository. This repository uses random forest imputation to fill in missing values allowing more data to be used to better train the LR Model. 
    """
)

#st.write("Dataset Comparison")
df_csv = pd.read_csv('data/renamed_num.csv')
df_csv['Legend'] = 'CSV Data'

# checking user data for existing
if "new_df" in st.session_state and not st.session_state.new_df.empty:
    df_user = st.session_state.new_df.copy()
    df_user['Legend'] = 'User Input'
    combined_df = pd.concat([df_csv, df_user], ignore_index=True)
    #st.dataframe(combined_df)
else:
    combined_df = df_csv
#differenciation user data vs ds
color_map = {"CSV Data": "purple", "User Input": "pink"}

x_scatter = st.selectbox('Scatter Plot - X axis', combined_df.columns, key='scatter_x')
y_scatter = st.selectbox('Scatter Plot - Y axis', combined_df.columns, key='scatter_y')
fig_scatter = px.scatter(
    combined_df,
    x=x_scatter,
    y=y_scatter,
    color="Legend",
    color_discrete_map=color_map,
    title="Scatter Plot: User condotion compared to CSV Data"
)

st.plotly_chart(fig_scatter)



# Histogram (for numeric variables)
st.write("## Histogram")
st.write("### Select the variable to plot")
numeric_cols = combined_df.select_dtypes(include='number').columns
hist_var = st.selectbox('Histogram Variable', numeric_cols, key='hist_var')
fig_hist = px.histogram(
    combined_df, 
    x=hist_var, 
    color="Legend", 
    color_discrete_map=color_map,
    barmode="overlay",
    opacity=0.75,
    title="Histogram (CSV Data and User Input)"
)
st.plotly_chart(fig_hist)

# Boxplot
st.write("## Boxplot")
st.write("### Select the variables to plot")
x_box = st.selectbox('Boxplot - X axis', combined_df.columns, key='box_x')
y_box = st.selectbox('Boxplot - Y axis', combined_df.columns, key='box_y')
fig_box = px.box(
    combined_df, 
    x=x_box, 
    y=y_box, 
    color="Legend", 
    color_discrete_map=color_map,
    title="Boxplot (CSV Data and User Input)"
)
st.plotly_chart(fig_box)

# Correlation Heatmap
st.write("## Correlation Heatmap")
fig_corr = px.imshow(
    combined_df.select_dtypes(include='number').corr(), 
    text_auto=True,
    title="Correlation Heatmap (CSV Data and User Input)"
)
st.plotly_chart(fig_corr)

