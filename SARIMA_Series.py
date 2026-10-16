import pandas as pd
import numpy as np
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

train = pd.read_csv("regresion_train.csv")
validacion = pd.read_csv("regresion_validacion.csv")
test = pd.read_csv("regresion_test.csv")

print("Train:", train.shape)
print("Validación:", validacion.shape)
print("Test:", test.shape)

#convertir fecha
for df in [train, validacion, test]:
    df["fecha_hora"] = pd.to_datetime(
        df["fecha_hora"],
        errors="coerce"
    )

#crear la serie para SARIMA
train_sarima = (
    train[["fecha_hora", "comuna", "ghi"]]
    .dropna()
    .sort_values(["comuna", "fecha_hora"])
    .reset_index(drop=True)
)

validacion_sarima = (
    validacion[["fecha_hora", "comuna", "ghi"]]
    .dropna()
    .sort_values(["comuna", "fecha_hora"])
    .reset_index(drop=True)
)

test_sarima = (
    test[["fecha_hora", "comuna", "ghi"]]
    .dropna()
    .sort_values(["comuna", "fecha_hora"])
    .reset_index(drop=True)
)

#verificar datos
for nombre, df in [
    ("Entrenamiento", train_sarima),
    ("Validación", validacion_sarima),
    ("Prueba", test_sarima)
]:
    print(f"\n{nombre}")

    for comuna, grupo in df.groupby("comuna"):
        print(
            comuna,
            "| registros:", len(grupo),
            "| periodo:", grupo["fecha_hora"].min(),
            "a", grupo["fecha_hora"].max()
        )

#definir configuraciones
configuraciones = [
    {
        "Configuración": "SARIMA 1",
        "order": (1, 0, 0),
        "seasonal_order": (1, 0, 0, 24)
    },
    {
        "Configuración": "SARIMA 2",
        "order": (1, 0, 1),
        "seasonal_order": (1, 0, 0, 24)
    }
]

#evaluar en validación
import gc
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

resultados_validacion_sarima = []
resultados_por_sitio = []

for config in configuraciones:

    reales_globales = []
    predicciones_globales = []

    print(f"\nEvaluando {config['Configuración']}")

    for comuna in sorted(train_sarima["comuna"].unique()):

        y_train = (
            train_sarima.loc[
                train_sarima["comuna"] == comuna,
                "ghi"
            ]
            .to_numpy(dtype="float64")
        )

        y_valid = (
            validacion_sarima.loc[
                validacion_sarima["comuna"] == comuna,
                "ghi"
            ]
            .to_numpy(dtype="float64")
        )

        modelo = SARIMAX(
            y_train,
            order=config["order"],
            seasonal_order=config["seasonal_order"],
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
            concentrate_scale=True
        )

        # Ahorrar memoria, pero conservar los estados que necesita extend()
        modelo.ssm.set_conserve_memory(
            memory_no_forecast_cov=True,
            memory_no_filtered=True,
            memory_no_gain=True,
            memory_no_smoothing=True,
            memory_no_std_forecast=True
        )

        ajuste = modelo.fit(
            disp=False,
            maxiter=30,
            cov_type="none"
        )

        # Continuar la serie con los datos reales de validación
        ajuste_validacion = ajuste.extend(y_valid)

        pred_valid = np.asarray(
            ajuste_validacion.fittedvalues,
            dtype=float
        )

        pred_valid = np.clip(pred_valid, 0, None)

        mae = mean_absolute_error(y_valid, pred_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
        r2 = r2_score(y_valid, pred_valid)

        resultados_por_sitio.append({
            "Configuración": config["Configuración"],
            "Localidad": comuna,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        reales_globales.extend(y_valid)
        predicciones_globales.extend(pred_valid)

        print(
            comuna,
            "| MAE:", round(mae, 2),
            "| RMSE:", round(rmse, 2)
        )

        # Liberar memoria antes de pasar al siguiente sitio
        del modelo
        del ajuste
        del ajuste_validacion
        del y_train
        del y_valid
        del pred_valid

        gc.collect()

    resultados_validacion_sarima.append({
        "Configuración": config["Configuración"],
        "Orden": str(config["order"]),
        "Orden estacional": str(config["seasonal_order"]),
        "MAE": mean_absolute_error(
            reales_globales,
            predicciones_globales
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                reales_globales,
                predicciones_globales
            )
        ),
        "R2": r2_score(
            reales_globales,
            predicciones_globales
        )
    })

    del reales_globales
    del predicciones_globales
    gc.collect()

#mostrar en tabla
tabla_validacion_sarima = pd.DataFrame(
    resultados_validacion_sarima
).sort_values("MAE")

tabla_sarima_por_sitio = pd.DataFrame(
    resultados_por_sitio
)

print(tabla_validacion_sarima)
print(tabla_sarima_por_sitio)

#prueba final
import gc

config_final = {
    "order": (1, 0, 0),
    "seasonal_order": (1, 0, 0, 24)
}

train_final_sarima = pd.concat(
    [train_sarima, validacion_sarima],
    ignore_index=True
).sort_values(["comuna", "fecha_hora"])

resultados_test_sarima = []

for comuna in sorted(train_final_sarima["comuna"].unique()):

    y_train_final = train_final_sarima.loc[
        train_final_sarima["comuna"] == comuna,
        "ghi"
    ].to_numpy(dtype=float)

    grupo_test = (
        test_sarima[test_sarima["comuna"] == comuna]
        .sort_values("fecha_hora")
        .reset_index(drop=True)
    )

    y_test_local = grupo_test["ghi"].to_numpy(dtype=float)

    modelo = SARIMAX(
        y_train_final,
        order=config_final["order"],
        seasonal_order=config_final["seasonal_order"],
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
        concentrate_scale=True
    )

    modelo.ssm.set_conserve_memory(
        memory_no_forecast_cov=True,
        memory_no_filtered=True,
        memory_no_gain=True,
        memory_no_smoothing=True,
        memory_no_std_forecast=True
    )

    ajuste = modelo.fit(
        disp=False,
        maxiter=30,
        cov_type="none"
    )

    extension = ajuste.extend(y_test_local)

    predicciones = np.asarray(
        extension.fittedvalues,
        dtype=float
    )

    # Se elimina la primera observación para evaluar
    # exactamente el pronóstico de la hora siguiente
    reales = y_test_local[1:]
    predicciones = np.clip(predicciones[1:], 0, None)
    fechas = grupo_test["fecha_hora"].iloc[1:].to_numpy()

    for fecha, real, prediccion in zip(
        fechas,
        reales,
        predicciones
    ):
        resultados_test_sarima.append({
            "fecha_hora": fecha,
            "comuna": comuna,
            "ghi_real": real,
            "ghi_pronosticado": prediccion
        })

    del modelo, ajuste, extension
    gc.collect()

resultados_test_sarima = pd.DataFrame(
    resultados_test_sarima
)

print(resultados_test_sarima.shape)
print(resultados_test_sarima.head())

mae_sarima = mean_absolute_error(
    resultados_test_sarima["ghi_real"],
    resultados_test_sarima["ghi_pronosticado"]
)

rmse_sarima = np.sqrt(
    mean_squared_error(
        resultados_test_sarima["ghi_real"],
        resultados_test_sarima["ghi_pronosticado"]
    )
)

r2_sarima = r2_score(
    resultados_test_sarima["ghi_real"],
    resultados_test_sarima["ghi_pronosticado"]
)

print("SARIMA FINAL")
print(f"MAE:  {mae_sarima:.4f} W/m²")
print(f"RMSE: {rmse_sarima:.4f} W/m²")
print(f"R²:   {r2_sarima:.4f}")

metricas_sarima_localidad = []

for comuna, grupo in resultados_test_sarima.groupby("comuna"):

    metricas_sarima_localidad.append({
        "Localidad": comuna,
        "MAE": mean_absolute_error(
            grupo["ghi_real"],
            grupo["ghi_pronosticado"]
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                grupo["ghi_real"],
                grupo["ghi_pronosticado"]
            )
        ),
        "R2": r2_score(
            grupo["ghi_real"],
            grupo["ghi_pronosticado"]
        )
    })

metricas_sarima_localidad = pd.DataFrame(
    metricas_sarima_localidad
)

print(metricas_sarima_localidad)

#graficos
import matplotlib.pyplot as plt

for comuna in resultados_test_sarima["comuna"].unique():

    muestra = (
        resultados_test_sarima[
            resultados_test_sarima["comuna"] == comuna
        ]
        .sort_values("fecha_hora")
        .head(24 * 7)
    )

    plt.figure(figsize=(14, 5))

    plt.plot(
        muestra["fecha_hora"],
        muestra["ghi_real"],
        label="GHI real",
        linewidth=2
    )

    plt.plot(
        muestra["fecha_hora"],
        muestra["ghi_pronosticado"],
        label="GHI pronosticado",
        linestyle="--",
        linewidth=2
    )

    plt.title(f"Valores reales y pronosticados de GHI en {comuna} — SARIMA")
    plt.xlabel("Fecha y hora")
    plt.ylabel("GHI (W/m²)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        f"sarima_real_vs_pronosticado_{comuna}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

resultados_test_sarima["residuo"] = (
    resultados_test_sarima["ghi_real"]
    - resultados_test_sarima["ghi_pronosticado"]
)

print(
    "Media de residuos:",
    resultados_test_sarima["residuo"].mean()
)

print(
    "Desviación estándar:",
    resultados_test_sarima["residuo"].std()
)

#guardar modelo
from google.colab import drive
import os
import joblib
import json

drive.mount("/content/drive")

ruta_modelos = "/content/drive/MyDrive/modelos_radiacion"
os.makedirs(ruta_modelos, exist_ok=True)
