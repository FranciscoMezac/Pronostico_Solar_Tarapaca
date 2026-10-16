#importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#definir archivos e información de ubicación
sitios = {
    "Pozo Almonte": {
        "archivo": "/content/pozo_almonte.csv",
        "latitud": -20.26753140414636,
        "longitud": -69.77801103385463,
        "altitud": 1037.0
    },
    "Pica": {
        "archivo": "/content/pica.csv",
        "latitud": -20.49088377081416,
        "longitud": -69.33031816276088,
        "altitud": 1332.0
    },
    "Colchane": {
        "archivo": "/content/colchane.csv",
        "latitud": -19.698839534259942,
        "longitud": -68.90281677246094,
        "altitud": 4161.0
    }
}

#cargar y unir lo 3 archivos
bases = []

for nombre_sitio, info in sitios.items():
    df = pd.read_csv(info["archivo"], skiprows=41)

    df["Fecha/Hora"] = pd.to_datetime(df["Fecha/Hora"])
    df = df.sort_values("Fecha/Hora")

    df["sitio"] = nombre_sitio
    df["latitud"] = info["latitud"]
    df["longitud"] = info["longitud"]
    df["altitud"] = info["altitud"]

    bases.append(df)

df_total = pd.concat(bases, ignore_index=True)

print(df_total.head())

#verificar carga de datos
print("Cantidad de filas y columnas:")
print(df_total.shape)

print("\nCantidad de registros por sitio:")
print(df_total["sitio"].value_counts())

print("\nPeriodo disponible:")
print(df_total["Fecha/Hora"].min())
print(df_total["Fecha/Hora"].max())

print("\nColumnas:")
print(df_total.columns.tolist())

print("\nValores faltantes por columna:")
print(df_total.isna().sum())

#revisar primeras y ultimas columnas
print(df_total.head(10))

print(df_total.tail(10))

print("Tamaño de la base:")
print(df_total.shape)

print("\nValores faltantes por columna:")
print(df_total.isna().sum())

print("\nRegistros por sitio:")
print(df_total["sitio"].value_counts())

#revisión de la estructura de los datos
print("Tamaño de la base:")
print(df_total.shape)

print("\nRegistros por sitio:")
print(df_total["sitio"].value_counts())

print("\nPeriodo por sitio:")
print(df_total.groupby("sitio")["Fecha/Hora"].agg(["min", "max", "count"]))

print("\nValores faltantes por columna:")
print(df_total.isna().sum())

#evolución temporal de la radiación solar global horizontal.
# Crear promedio mensual general de GHI considerando todos los sitios
df_total["anio_mes"] = df_total["Fecha/Hora"].dt.to_period("M").dt.to_timestamp()

ghi_mensual_total = df_total.groupby("anio_mes")["ghi"].mean().reset_index()

# Gráfico de evolución temporal
plt.figure(figsize=(14,5))

plt.plot(
    ghi_mensual_total["anio_mes"],
    ghi_mensual_total["ghi"]
)

plt.title("Evolución temporal de la radiación solar global horizontal (GHI)")
plt.xlabel("Fecha")
plt.ylabel("GHI promedio mensual [W/m²]")
plt.grid(True, alpha=0.3)
plt.show()

#promedio mensual de radiación por año
df_total["anio_mes"] = df_total["Fecha/Hora"].dt.to_period("M").dt.to_timestamp()

ghi_mensual = df_total.groupby(["anio_mes", "sitio"])["ghi"].mean().reset_index()

plt.figure(figsize=(14,5))

for sitio in ghi_mensual["sitio"].unique():
    datos = ghi_mensual[ghi_mensual["sitio"] == sitio]
    plt.plot(datos["anio_mes"], datos["ghi"], label=sitio)

plt.title("Promedio mensual de radiación solar global horizontal (GHI)")
plt.xlabel("Fecha")
plt.ylabel("GHI promedio mensual [W/m²]")
plt.legend()
plt.show()

#promedio de radiación solar por hora
df_total["hora"] = df_total["Fecha/Hora"].dt.hour
ghi_horario = df_total.groupby(["hora", "sitio"])["ghi"].mean().reset_index()

plt.figure(figsize=(10,5))

for sitio in ghi_horario["sitio"].unique():
    datos = ghi_horario[ghi_horario["sitio"] == sitio]
    plt.plot(datos["hora"], datos["ghi"], marker="o", label=sitio)

plt.title("Promedio horario de radiación solar global horizontal (GHI)")
plt.xlabel("Hora del día")
plt.ylabel("GHI promedio [W/m²]")
plt.xticks(range(0, 24))
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#distribución ghi por sitio (global)
plt.figure(figsize=(8,5))

df_total.boxplot(column="ghi", by="sitio", grid=False)

plt.title("Distribución de GHI por sitio")
plt.suptitle("")
plt.xlabel("Sitio")
plt.ylabel("GHI [W/m²]")
plt.show()

#ghi por sitio (solo horas solares)
df_dia = df_total[df_total["ghi"] > 0].copy()

plt.figure(figsize=(8,5))

df_dia.boxplot(column="ghi", by="sitio", grid=False)

plt.title("Distribución de GHI por sitio durante horas con radiación")
plt.suptitle("")
plt.xlabel("Sitio")
plt.ylabel("GHI [W/m²]")
plt.show()

#promedio mensual por sitio
df_total["mes"] = df_total["Fecha/Hora"].dt.month

ghi_mes = df_total.groupby(["mes", "sitio"])["ghi"].mean().reset_index()

plt.figure(figsize=(10,5))

for sitio in ghi_mes["sitio"].unique():
    datos = ghi_mes[ghi_mes["sitio"] == sitio]
    plt.plot(datos["mes"], datos["ghi"], marker="o", label=sitio)

plt.title("Promedio mensual general de GHI por sitio")
plt.xlabel("Mes")
plt.ylabel("GHI promedio [W/m²]")
plt.xticks(range(1, 13))
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#matriz de correlación
import seaborn as sns

variables_corr = ["ghi", "temp", "vel", "cloud", "shadow", "latitud", "longitud", "altitud"]

corr = df_total[variables_corr].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación entre variables")
plt.show()

#ghi promedio según nubosidad y sitio
ghi_cloud = df_total.groupby(["cloud", "sitio"])["ghi"].mean().reset_index()

plt.figure(figsize=(8,5))

for sitio in ghi_cloud["sitio"].unique():
    datos = ghi_cloud[ghi_cloud["sitio"] == sitio]
    plt.bar(
        datos["cloud"].astype(str) + " - " + sitio,
        datos["ghi"]
    )

plt.title("GHI promedio según nubosidad por sitio")
plt.xlabel("Cloud - Sitio")
plt.ylabel("GHI promedio [W/m²]")
plt.xticks(rotation=45)
plt.show()

#revisar valores imposibles
print(df_total.groupby("sitio")["ghi"].describe())

print("Mínimo GHI:", df_total["ghi"].min())
print("Máximo GHI:", df_total["ghi"].max())

#revisar temperatura y viento
print(df_total[["ghi", "temp", "vel"]].describe())

#revisar outliers
variables_revisar = ["ghi", "temp", "vel", "cloud", "shadow"]

print("Resumen estadístico:")
print(df_total[variables_revisar].describe())

print("\nMínimos por sitio:")
print(df_total.groupby("sitio")[variables_revisar].min())

print("\nMáximos por sitio:")
print(df_total.groupby("sitio")[variables_revisar].max())

#guardar modelo
from google.colab import drive
import os
import joblib
import json

drive.mount("/content/drive")

ruta_modelos = "/content/drive/MyDrive/modelos_radiacion"
os.makedirs(ruta_modelos, exist_ok=True)

joblib.dump(
    modelo_regresion_final,
    f"{ruta_modelos}/regresion_lineal_final.joblib"
)
