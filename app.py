import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



st.set_page_config(
    page_title="App User Behavior Dashboard",
    layout="wide"
)


@st.cache_data

def load_data():
    return pd.read_csv("app_user_behaviour_cleaned.csv")


df = load_data()



st.title("📊 App User Behavior Segmentation Dashboard")

st.markdown("---")

st.sidebar.header("Filters")

selected_device = st.sidebar.multiselect(
    "Select Device Type",
    options=df['device_type'].unique(),
    default=df['device_type'].unique()
)

selected_subscription = st.sidebar.multiselect(
    "Select Subscription Type",
    options=df['subscription_type'].unique(),
    default=df['subscription_type'].unique()
)


filtered_df = df[
    (df['device_type'].isin(selected_device)) &
    (df['subscription_type'].isin(selected_subscription))
]  



st.subheader("📌 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Users", len(filtered_df))

with col2:
    st.metric(
        "Average Engagement",
        round(filtered_df['engagement_score'].mean(),2)
    )

with col3:
    st.metric(
        "Average Session Duration",
        round(filtered_df['avg_session_duration_min'].mean(),2)
    )

with col4:
    st.metric(
        "Average Churn Risk",
        round(filtered_df['churn_risk_score'].mean(),2)
    )
st.markdown("---")


st.subheader("📊 Cluster Distribution")

cluster_count = filtered_df['User_Group'].value_counts()

fig1 = px.bar(
    x=cluster_count.index,
    y=cluster_count.values,
    labels={'x':'User Group','y':'Count'},
    title="Users in Each Cluster"
)

st.plotly_chart(fig1, use_container_width=True)


fig2 = px.pie(
    names=cluster_count.index,
    values=cluster_count.values,
    title="Cluster Percentage"
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

st.subheader("📈 Engagement Analysis")

fig3 = px.box(
    filtered_df,
    x='User_Group',
    y='engagement_score',
    color='User_Group',
    title="Engagement Score by Cluster"
)

st.plotly_chart(fig3, use_container_width=True)



fig4 = px.box(
    filtered_df,
    x='User_Group',
    y='avg_session_duration_min',
    color='User_Group',
    title="Session Duration by Cluster"
)

st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")



st.subheader("⚠️ Churn Risk Analysis")

fig5 = px.box(
    filtered_df,
    x='User_Group',
    y='churn_risk_score',
    color='User_Group',
    title="Churn Risk by Cluster"
)

st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")



st.subheader("🔥 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(12,8))

corr = filtered_df.corr(numeric_only=True)

sns.heatmap(corr, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.markdown("---")



st.subheader("📋 Cluster Profile")

profile = filtered_df.groupby('User_Group')[[
    'sessions_per_week',
    'avg_session_duration_min',
    'engagement_score',
    'churn_risk_score'
]].mean()

st.dataframe(profile)

st.markdown("---")

st.subheader("🗂️ Dataset Preview")

st.dataframe(filtered_df.head())