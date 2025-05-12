import cvxpy as cp
import numpy as np
import pandas as pd
from io import BytesIO

def ejecutar_optimizacion(X, mes_inicio):
    # ----------------------------
    # Parámetros del modelo
    # ----------------------------
    Y = 1.5 * X
    uplift_max = 4_000_000_000  # Límite superior del uplift

    growth_rates = np.array([0.10, -0.05, 0.08, 0.00,
                             -0.10, 0.12, 0.15, 0.05,
                             -0.02, 0.04, 0.07, 0.11])
    b = 1 + growth_rates
    w_month = b / b.sum()
    Y_month = Y * w_month

    channel_weights = np.array([0.14, 0.36, 0.07, 0.16, 0.28])
    channel_names = ['Droguerías Cadena', 'Droguerías Independientes',
                     'Superetes', 'Supermercados', 'Tradicional']
    J = len(channel_weights)
    Y_mj = np.outer(Y_month, channel_weights)

    medios = ['inv_dig_display', 'inv_dig_e-commerce', 'inv_dig_search', 'inv_dig_social',
              'inv_dig_video', 'inv_tv', 'inv_radio', 'inv_OOH']
    I, M = len(medios), 12
    coef_medio = np.array([1.1, 1.12, 2.07, 2.09, 2.05, 1.09, 1.0, 1.05])
    coef_mes = np.array([0.98, 1.5, 1.52, 1.5, 1.8, 1.37,
                         1.71, 1.37, 0.82, 1.08, 0.95, 1.01])

    c = np.zeros((M, I, J))
    for m in range(mes_inicio, M):
        for i in range(I):
            for j in range(J):
                c[m, i, j] = coef_medio[i] * coef_mes[m] * channel_weights[j]

    porcentajes_maximos = {
        'inv_dig_display': 0.15,
        'inv_dig_e-commerce': 0.12,
        'inv_dig_search': 0.14,
        'inv_dig_social': 0.13,
        'inv_dig_video': 0.15,
        'inv_tv': 0.48,
        'inv_radio': 0.10,
        'inv_OOH': 0.08,
    }

    # ----------------------------
    # Variables y restricciones
    # ----------------------------
    x = cp.Variable((I, M), nonneg=True)
    constraints = [cp.sum(x) >= 0.99 * X, cp.sum(x) <= X]

    for i, medio in enumerate(medios):
        for m in range(M):
            if m < mes_inicio:
                constraints.append(x[i, m] == 0)

        constraints.append(cp.sum(x[i, :]) <= porcentajes_maximos[medio] * X)
        constraints.append(cp.sum(x[i, :]) >= 0.01 * X)

    uplift_total_estimado = cp.sum([
        cp.sum(cp.multiply(c[m, :, j], x[:, m]))
        for m in range(mes_inicio, M)
        for j in range(J)
    ])
    constraints.append(uplift_total_estimado <= uplift_max)

    # ----------------------------
    # Función objetivo
    # ----------------------------
    expr = 0
    for m in range(mes_inicio, M):
        for j in range(J):
            pred = cp.sum(cp.multiply(c[m, :, j], x[:, m]))
            expr += cp.square(pred - Y_mj[m, j])

    prob = cp.Problem(cp.Minimize(expr), constraints)
    prob.solve()

    # ----------------------------
    # Resultados
    # ----------------------------
    if x.value is not None:
        meses_etiquetas = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                           "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
        df_inv = pd.DataFrame(x.value, index=medios, columns=meses_etiquetas)

        uplift_pred = np.zeros((J, M))
        for m in range(mes_inicio, M):
            for j in range(J):
                uplift_pred[j, m] = np.dot(c[m, :, j], x.value[:, m])
        df_uplift = pd.DataFrame(uplift_pred, index=channel_names, columns=meses_etiquetas)

        # ROI por medio
        total_inversion_medio = df_inv.sum(axis=1)
        uplift_medio = np.zeros(I)
        for i in range(I):
            for m in range(mes_inicio, M):
                uplift_medio[i] += x.value[i, m] * np.sum(c[m, i, :])
        roi_medio = uplift_medio / total_inversion_medio
        df_roi = pd.Series(roi_medio, index=medios)

        # Exportar a Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_inv.to_excel(writer, sheet_name='Inversion')
            df_uplift.to_excel(writer, sheet_name='Uplift_Predicho')
            pd.DataFrame({'ROI': df_roi}).to_excel(writer, sheet_name='ROI_por_medio')
        excel_data = output.getvalue()

        return df_inv, df_uplift, df_roi, excel_data
    else:
        raise ValueError("No se encontró solución factible.")
