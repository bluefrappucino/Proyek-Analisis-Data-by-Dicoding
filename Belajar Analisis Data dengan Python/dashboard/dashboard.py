import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

# Load data
@st.cache
def load_data():
    df_day = pd.read_csv("day.csv")
    df_hour = pd.read_csv("hour.csv")
    return df_day, df_hour

def main():
    df_day, df_hour = load_data()

    st.title("Bike Sharing Dataset Dashboard")
    st.sidebar.title("Dashboard Options")
    option = st.sidebar.selectbox("Select Option", ("Summer Rental Increase", "Peak Hour Analysis", "Weekday vs Weekend Rental", "Temperature Impact", "Hourly Rental Average", "Predicted Rental Increase"))

    if option == "Summer Rental Increase":
        st.subheader("Summer Rental Increase Analysis")

        # Calculate average rental per season
        df_season_avg = df_day.groupby("season")["cnt"].mean()

        # Calculate percentage increase during summer
        summer_avg = df_season_avg[2]
        other_seasons_avg = df_season_avg.drop(2).mean()
        increase_pct = ((summer_avg - other_seasons_avg) / other_seasons_avg) * 100

        st.write(f"Increase in rental during summer: {increase_pct:.2f}%")

    elif option == "Peak Hour Analysis":
        st.subheader("Peak Hour Analysis")

        # Calculate average rental per hour
        df_hourly = df_hour.groupby("hr")["cnt"].mean()

        # Find peak hour
        peak_hour = df_hourly.idxmax()

        st.write(f"The peak hour for rentals is at {peak_hour} o'clock.")

        # Plot hourly rental average
        st.write("### Hourly Rental Average")
        fig = px.line(df_hourly.reset_index(), x='hr', y='cnt', title='Rata-rata Jumlah Sewa per Jam')
        fig.update_layout(xaxis_title='Jam', yaxis_title='Rata-rata Jumlah Sewa')
        st.plotly_chart(fig)

    elif option == "Weekday vs Weekend Rental":
        st.subheader("Weekday vs Weekend Rental Analysis")

        # Calculate weekday vs weekend rental
        t_stat, p_value = stats.ttest_ind(df_day[df_day["weekday"] <= 4]["cnt"], df_day[df_day["weekday"] >= 5]["cnt"])

        if p_value < 0.05:
            st.write("There is a significant difference in rentals between weekdays and weekends (p-value < 0.05).")
        else:
            st.write("There is no significant difference in rentals between weekdays and weekends (p-value >= 0.05).")

        # Plot weekday rental average
        st.write("### Weekday Rental Average")
        df_weekday_avg = df_day.groupby("weekday")["cnt"].mean().reset_index()
        df_weekday_avg['weekday'] = df_weekday_avg['weekday'].map({0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'})
        fig = px.bar(df_weekday_avg, x='weekday', y='cnt', title='Rata-Rata Jumlah Sewa per Hari', labels={'weekday': 'Hari', 'cnt': 'Rata-Rata Jumlah Sewa'})
        fig.update_traces(marker_color='skyblue')
        fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']}, 
                          yaxis_title='Rata-Rata Jumlah Sewa', xaxis_title='Hari', showlegend=False)
        st.plotly_chart(fig)

    elif option == "Temperature Impact":
        st.subheader("Temperature Impact Analysis")

        # Model training and evaluation
        X_train, X_test, y_train, y_test = train_test_split(df_day[["temp"]], df_day["cnt"], test_size=0.25)
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        # Prediction
        predicted_cnt = model.predict([[1]])
        increase_cnt = predicted_cnt[0] - y_test.mean()

        st.write(f"R-squared: {score:.2f}")
        st.write(f"Predicted increase in bike rentals: {increase_cnt:.2f}")

        # Plot scatter plot and regression line
        fig1 = px.scatter(x=X_test["temp"], y=y_test, title="Scatter Plot: Suhu vs Jumlah Sewa", labels={"temp": "Suhu (Celcius)", "cnt": "Jumlah Sewa"})
        fig1.add_scatter(x=[1], y=[predicted_cnt[0]], mode='markers', name='Prediction', marker=dict(color='red', size=10, symbol='cross'))

        fig2 = px.scatter(x=X_test.squeeze(), y=y_test, title='Regresi Linear: Pengaruh Suhu terhadap Jumlah Sewa', labels={'x': 'Suhu (Celcius)', 'y': 'Jumlah Sewa'}, trendline="ols")
        fig2.update_traces(marker=dict(color='blue'), mode='markers', name='Data Uji')
        fig2.add_scatter(x=X_test.squeeze(), y=model.predict(X_test), mode='lines', line=dict(color='red', width=2), name='Regression Line')
        fig2.update_layout(showlegend=True)

        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    elif option == "Hourly Rental Average":
        st.subheader("Hourly Rental Average Analysis")

        # Calculate average rental per hour
        df_hourly = df_hour.groupby("hr")["cnt"].mean().reset_index()

        # Plot hourly rental average
        fig = px.line(df_hourly, x='hr', y='cnt', markers=True, title='Rata-rata Jumlah Sewa per Jam')
        fig.update_layout(xaxis_title='Jam', yaxis_title='Rata-rata Jumlah Sewa', xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig)

    elif option == "Predicted Rental Increase":
        st.subheader("Predicted Rental Increase Analysis")

        # Model training and evaluation
        X_train, X_test, y_train, y_test = train_test_split(df_day[["temp"]], df_day["cnt"], test_size=0.25)
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        # Prediction
        predicted_cnt = model.predict([[1]])
        increase_cnt = float(predicted_cnt[0])

        st.write(f"R-squared: {score:.2f}")
        st.write(f"Predicted increase in bike rentals: {increase_cnt:.2f}")

        # Plot scatter plot and regression line
        fig1 = px.scatter(x=X_test["temp"], y=y_test, title="Scatter Plot: Suhu vs Jumlah Sewa", labels={"temp": "Suhu (Celcius)", "cnt": "Jumlah Sewa"})
        fig1.add_scatter(x=[1], y=[predicted_cnt[0]], mode='markers', name='Prediction', marker=dict(color='red', size=10, symbol='cross'))

        fig2 = px.scatter(x=X_test.squeeze(), y=y_test, title='Regresi Linear: Pengaruh Suhu terhadap Jumlah Sewa', labels={'x': 'Suhu (Celcius)', 'y': 'Jumlah Sewa'}, trendline="ols")
        fig2.update_traces(marker=dict(color='blue'), mode='markers', name='Data Uji')
        fig2.add_scatter(x=X_test.squeeze(), y=model.predict(X_test), mode='lines', line=dict(color='red', width=2), name='Regression Line')
        fig2.update_layout(showlegend=True)

        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

if __name__ == '__main__':
    main()
