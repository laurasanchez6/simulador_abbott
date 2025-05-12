import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
from optimizador import ejecutar_optimizacion
from io import BytesIO

# ---------- LOGIN ----------
names = ['Laura Sánchez']
usernames = ['laura']
passwords = ['1234']
hashed_pw = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(
    names, usernames, hashed_pw,
    "app_login", "app_cookie_key", cookie_expiry_days=1
)

name, auth_status, username = authenticator.login("Login", "main")

if auth_status:
    st.title("Simulador de Optimización de Inversión en Medios")

    st.success(f"Bienvenida, {name}")

    # ---------- INPUTS ----------
    presupuesto = st.number_input("💰 Monto total a invertir (COP)", min_value=500_000_000, value=1_000_000_000, step=100_000_000)

    mes_nombre = st.selectbox("📆 Mes desde el cual iniciar la inversión", [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ])
    mes_inicio = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                  "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"].index(mes_nombre)

    if st.button("🚀 Ejecutar escenario"):
        df_inv, df_uplift, df_roi, excel_bytes = ejecutar_optimizacion(X=presupuesto, mes_inicio=mes_inicio)

        st.subheader("📊 Distribución de Inversión")
        st.dataframe(df_inv.style.format("{:,.0f}"))

        st.subheader("📈 Uplift Predicho por Canal")
        st.dataframe(df_uplift.style.format("{:,.0f}"))

        st.subheader("📉 ROI por Medio")
        st.bar_chart(df_roi)

        st.download_button(
            label="📥 Descargar Excel del Escenario",
            data=excel_bytes,
            file_name="escenario_optimizacion.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

elif auth_status == False:
    st.error("❌ Usuario o contraseña incorrectos")
elif auth_status is None:
    st.warning("🔐 Por favor, ingresa tus credenciales")
