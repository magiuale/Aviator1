# Aviator1
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title("Previsione moltiplicatori Aviator")
st.write("Carica un file CSV contenente i moltiplicatori passati")

uploaded_file = st.file_uploader("Scegli un file CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'multiplier' not in data.columns:
        st.error("Il file deve contenere una colonna chiamata 'multiplier'")
    else:
        st.success("Dati caricati con successo!")
        st.write(data.head())

        # Preparazione dei dati
        X = np.array(data.index).reshape(-1, 1)
        y = data['multiplier'].values

        # Divisione dei dati per training e testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Addestramento del modello
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Previsioni future
        future_index = np.array(range(len(data), len(data) + 10)).reshape(-1, 1)
        predictions = model.predict(future_index)

        # Mostra le previsioni
        st.write("Previsioni per i prossimi 10 turni:")
        st.write(predictions)

        # Visualizzazione del grafico
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['multiplier'], label="Storico", marker='o')
        plt.plot(range(len(data), len(data) + 10), predictions, label="Previsione", color="red", marker='x')
        plt.xlabel("Turni")
        plt.ylabel("Moltiplicatore")
        plt.legend()
        st.pyplot(plt)
