#importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

#leer archivos
train = pd.read_csv("regresion_train.csv")
validacion = pd.read_csv("regresion_validacion.csv")
test = pd.read_csv("regresion_test.csv")

print("Train:", train.shape)
print("Validación:", validacion.shape)
print("Test:", test.shape)

#convertir fecha
for df in [train, validacion, test]:
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

# Modelo Base

#definir variable objetivo
target = "ghi_t1"

#usaremos solamente ghi
#¿Qué tan bien puede pronosticarse la próxima hora
#utilizando solamente la radiación actual?
features_base = ["ghi"]

#preparar datos
X_train = train[features_base]
y_train = train[target]

X_valid = validacion[features_base]
y_valid = validacion[target]

#entrenar modelo
modelo_base = LinearRegression()
modelo_base.fit(X_train, y_train)

#pronosticar validación
pred_valid_base = modelo_base.predict(X_valid)

#evaluar
mae_base = mean_absolute_error(y_valid, pred_valid_base)
rmse_base = np.sqrt(mean_squared_error(y_valid, pred_valid_base))
r2_base = r2_score(y_valid, pred_valid_base)

print("Modelo base")
print(f"MAE:  {mae_base:.4f} W/m²")
print(f"RMSE: {rmse_base:.4f} W/m²")
print(f"R²:   {r2_base:.4f}")

# Modelo Temporal

#agregar antecedentes de ghi
features_temporales = [
    "ghi",
    "ghi_lag_1",
    "ghi_lag_24",
    "hora_sin",
    "hora_cos",
    "dia_anio_sin",
    "dia_anio_cos"
]

#entrenamiento del modelo con antecedentes de ghi y evaluación
X_train = train[features_temporales]
X_valid = validacion[features_temporales]

modelo_temporal = LinearRegression()
modelo_temporal.fit(X_train, y_train)

pred_valid_temporal = modelo_temporal.predict(X_valid)

mae_temporal = mean_absolute_error(y_valid, pred_valid_temporal)
rmse_temporal = np.sqrt(mean_squared_error(y_valid, pred_valid_temporal))
r2_temporal = r2_score(y_valid, pred_valid_temporal)

print("Modelo temporal")
print(f"MAE:  {mae_temporal:.4f} W/m²")
print(f"RMSE: {rmse_temporal:.4f} W/m²")
print(f"R²:   {r2_temporal:.4f}")

# Modelo Temporal y Meteorológico

#agregar condiciones meteorológicas
features_meteorologicas = [
    "ghi",
    "ghi_lag_1",
    "ghi_lag_24",
    "temp",
    "vel",
    "cloud",
    "hora_sin",
    "hora_cos",
    "dia_anio_sin",
    "dia_anio_cos"
]

#entrenamiento del modelo con condiciones meteorológicas y evaluación
X_train = train[features_meteorologicas]
X_valid = validacion[features_meteorologicas]

modelo_meteorologico = LinearRegression()
modelo_meteorologico.fit(X_train, y_train)

pred_valid_meteorologico = modelo_meteorologico.predict(X_valid)

mae_meteorologico = mean_absolute_error(
    y_valid,
    pred_valid_meteorologico
)

rmse_meteorologico = np.sqrt(
    mean_squared_error(y_valid, pred_valid_meteorologico)
)

r2_meteorologico = r2_score(
    y_valid,
    pred_valid_meteorologico
)

print("Modelo temporal + meteorológico")
print(f"MAE:  {mae_meteorologico:.4f} W/m²")
print(f"RMSE: {rmse_meteorologico:.4f} W/m²")
print(f"R²:   {r2_meteorologico:.4f}")

# Modelo Multisitio

features_multisitio = [
    "ghi",
    "ghi_lag_1",
    "ghi_lag_24",
    "temp",
    "vel",
    "cloud",
    "hora_sin",
    "hora_cos",
    "dia_anio_sin",
    "dia_anio_cos",
    "colchane",
    "pica"
]

X_train = train[features_multisitio]
X_valid = validacion[features_multisitio]

modelo_multisitio = LinearRegression()
modelo_multisitio.fit(X_train, y_train)

pred_valid_multisitio = modelo_multisitio.predict(X_valid)

mae_multisitio = mean_absolute_error(
    y_valid,
    pred_valid_multisitio
)

rmse_multisitio = np.sqrt(
    mean_squared_error(y_valid, pred_valid_multisitio)
)

r2_multisitio = r2_score(
    y_valid,
    pred_valid_multisitio
)

print("Modelo multisitio completo")
print(f"MAE:  {mae_multisitio:.4f} W/m²")
print(f"RMSE: {rmse_multisitio:.4f} W/m²")
print(f"R²:   {r2_multisitio:.4f}")

# Resultados

resultados_validacion = pd.DataFrame({
    "Modelo": [
        "Base",
        "Temporal",
        "Temporal + meteorológico",
        "Multisitio completo"
    ],
    "MAE": [
        mae_base,
        mae_temporal,
        mae_meteorologico,
        mae_multisitio
    ],
    "RMSE": [
        rmse_base,
        rmse_temporal,
        rmse_meteorologico,
        rmse_multisitio
    ],
    "R2": [
        r2_base,
        r2_temporal,
        r2_meteorologico,
        r2_multisitio
    ]
}).sort_values("RMSE")

print(resultados_validacion)

features_finales = [
    "ghi",
    "ghi_lag_1",
    "ghi_lag_24",
    "temp",
    "vel",
    "cloud",
    "hora_sin",
    "hora_cos",
    "dia_anio_sin",
    "dia_anio_cos",
    "colchane",
    "pica"
]

target = "ghi_t1"

# Modelo Final

#eliminar filas incompletas
columnas_necesarias = features_finales + [target]

train = train.dropna(
    subset=columnas_necesarias
).copy()

validacion = validacion.dropna(
    subset=columnas_necesarias
).copy()

test = test.dropna(
    subset=columnas_necesarias
).copy()

#comprobar
print("Train:", train.shape)
print("Validación:", validacion.shape)
print("Test:", test.shape)

print(
    "Faltantes train:",
    train[columnas_necesarias].isna().sum().sum()
)

print(
    "Faltantes validación:",
    validacion[columnas_necesarias].isna().sum().sum()
)

print(
    "Faltantes test:",
    test[columnas_necesarias].isna().sum().sum()
)

#unir entrenamiento y validación
train_final = pd.concat(
    [train, validacion],
    ignore_index=True
)

train_final = train_final.sort_values(
    ["comuna", "fecha_hora"]
).reset_index(drop=True)

test = test.sort_values(
    ["comuna", "fecha_hora"]
).reset_index(drop=True)

print("Entrenamiento final:", train_final.shape)
print("Test final:", test.shape)

#preparar las entradas y el objetivo
X_train_final = train_final[features_finales]
y_train_final = train_final[target]

X_test = test[features_finales]
y_test = test[target]

#validación final
print("Faltantes X train:", X_train_final.isna().sum().sum())
print("Faltantes y train:", y_train_final.isna().sum())
print("Faltantes X test:", X_test.isna().sum().sum())
print("Faltantes y test:", y_test.isna().sum())

#entrenar regresion final
modelo_regresion_final = LinearRegression()

modelo_regresion_final.fit(
    X_train_final,
    y_train_final
)

print("Modelo entrenado correctamente.")

#generar predicciones en test
pred_test_original = modelo_regresion_final.predict(
    X_test
)

#ghi no puede ser negativo, se aplica restricción fisica
pred_test = np.maximum(
    pred_test_original,
    0
)

#calcular métricas finales
mae_test = mean_absolute_error(
    y_test,
    pred_test
)

rmse_test = np.sqrt(
    mean_squared_error(
        y_test,
        pred_test
    )
)

r2_test = r2_score(
    y_test,
    pred_test
)

print("REGRESIÓN LINEAL MULTISITIO FINAL")
print(f"MAE:  {mae_test:.4f} W/m²")
print(f"RMSE: {rmse_test:.4f} W/m²")
print(f"R²:   {r2_test:.4f}")

#revisar predicciones negativas
cantidad_negativas = np.sum(
    pred_test_original < 0
)

porcentaje_negativas = (
    cantidad_negativas
    / len(pred_test_original)
    * 100
)

print("Predicciones negativas originales:", cantidad_negativas)
print(f"Porcentaje: {porcentaje_negativas:.2f}%")
print("Predicción mínima original:", pred_test_original.min())

#tabla de resultados
resultados_test = test[
    ["fecha_hora", "comuna", target]
].copy()

resultados_test["prediccion_original"] = (
    pred_test_original
)

resultados_test["prediccion_ajustada"] = (
    pred_test
)

resultados_test["residuo"] = (
    resultados_test[target]
    - resultados_test["prediccion_ajustada"]
)

resultados_test["error_absoluto"] = (
    resultados_test["residuo"].abs()
)

print(resultados_test.head())

#métricas por comuna
metricas_comuna = []

for comuna, grupo in resultados_test.groupby("comuna"):

    mae = mean_absolute_error(
        grupo[target],
        grupo["prediccion_ajustada"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            grupo[target],
            grupo["prediccion_ajustada"]
        )
    )

    r2 = r2_score(
        grupo[target],
        grupo["prediccion_ajustada"]
    )

    metricas_comuna.append({
        "Comuna": comuna,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Registros": len(grupo)
    })

metricas_comuna = pd.DataFrame(
    metricas_comuna
)

print(metricas_comuna)

#graficar semana por columna
for comuna in resultados_test["comuna"].unique():

    muestra = (
        resultados_test[
            resultados_test["comuna"] == comuna
        ]
        .sort_values("fecha_hora")
        .head(24 * 7)
    )

    plt.figure(figsize=(14, 5))

    plt.plot(
        muestra["fecha_hora"],
        muestra[target],
        label="GHI real"
    )

    plt.plot(
        muestra["fecha_hora"],
        muestra["prediccion_ajustada"],
        label="GHI pronosticado"
    )

    plt.xlabel("Fecha y hora")
    plt.ylabel("GHI (W/m²)")
    plt.title(
        f"Regresión lineal multisitio — {comuna}"
    )

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#analizar residuos
residuos = (
    y_test.to_numpy()
    - pred_test
)

print("Media de residuos:", residuos.mean())
print("Desviación estándar:", residuos.std())

#graficar
plt.figure(figsize=(9, 5))

plt.scatter(
    pred_test,
    residuos,
    alpha=0.15
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("GHI pronosticado (W/m²)")
plt.ylabel("Residuo: real − pronosticado")
plt.title("Residuos de la regresión multisitio")
plt.tight_layout()
plt.show()

import joblib

#guardar modelo y resultados
joblib.dump(
    modelo_regresion_final,
    "modelo_regresion_multisitio_final.joblib"
)

resultados_test.to_csv(
    "resultados_regresion_multisitio_test.csv",
    index=False
)

metricas_comuna.to_csv(
    "metricas_regresion_por_comuna.csv",
    index=False
)

print("Modelo y resultados guardados.")
