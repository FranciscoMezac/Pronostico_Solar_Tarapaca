import pandas as pd
import numpy as np
import tensorflow as tf
import gc

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

tf.keras.utils.set_random_seed(42)

train = pd.read_csv("regresion_train.csv")
validacion = pd.read_csv("regresion_validacion.csv")
test = pd.read_csv("regresion_test.csv")

for df in [train, validacion, test]:
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

print("Train:", train.shape)
print("Validación:", validacion.shape)
print("Test:", test.shape)

features = [
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
columnas_necesarias = features + [target]

for df in [train, validacion, test]:
    df.dropna(subset=columnas_necesarias, inplace=True)
    df.sort_values(["comuna", "fecha_hora"], inplace=True)
    df.reset_index(drop=True, inplace=True)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

scaler_X.fit(train[features])
scaler_y.fit(train[[target]])

ventana = 24

def crear_secuencias(df_actual, df_historial=None):

    X_total = []
    y_total = []
    metadata_total = []

    for comuna in sorted(df_actual["comuna"].unique()):

        actual = (
            df_actual[df_actual["comuna"] == comuna]
            .sort_values("fecha_hora")
            .reset_index(drop=True)
        )

        if df_historial is None:

            combinado = actual.copy()

            X_escalado = scaler_X.transform(
                combinado[features]
            ).astype(np.float32)

            ventanas = sliding_window_view(
                X_escalado,
                window_shape=ventana,
                axis=0
            ).transpose(0, 2, 1)

            y = actual[target].to_numpy(
                dtype=np.float32
            )[ventana - 1:]

            metadata = actual[
                ["fecha_hora", "comuna"]
            ].iloc[ventana - 1:].copy()

        else:

            historial = (
                df_historial[
                    df_historial["comuna"] == comuna
                ]
                .sort_values("fecha_hora")
                .tail(ventana - 1)
            )

            combinado = pd.concat(
                [historial, actual],
                ignore_index=True
            )

            X_escalado = scaler_X.transform(
                combinado[features]
            ).astype(np.float32)

            ventanas = sliding_window_view(
                X_escalado,
                window_shape=ventana,
                axis=0
            ).transpose(0, 2, 1)

            y = actual[target].to_numpy(
                dtype=np.float32
            )

            metadata = actual[
                ["fecha_hora", "comuna"]
            ].copy()

        X_total.append(ventanas)
        y_total.append(y)
        metadata_total.append(metadata)

    return (
        np.concatenate(X_total),
        np.concatenate(y_total),
        pd.concat(metadata_total, ignore_index=True)
    )

X_train_seq, y_train_seq, meta_train = crear_secuencias(
    train
)

X_valid_seq, y_valid_seq, meta_valid = crear_secuencias(
    validacion,
    df_historial=train
)

X_test_seq, y_test_seq, meta_test = crear_secuencias(
    test,
    df_historial=validacion
)

y_train_scaled = scaler_y.transform(
    y_train_seq.reshape(-1, 1)
).ravel()

y_valid_scaled = scaler_y.transform(
    y_valid_seq.reshape(-1, 1)
).ravel()

print("Train:", X_train_seq.shape)
print("Validación:", X_valid_seq.shape)
print("Test:", X_test_seq.shape)

configuraciones_lstm = {
    "LSTM 1": 32,
    "LSTM 2": 64
}
#LSTM 1: 32 unidades.
#LSTM 2: 64 unidades.

#crear y validar modelos
resultados_validacion_lstm = []

for nombre, unidades in configuraciones_lstm.items():

    print(f"\nEntrenando {nombre}")

    tf.keras.backend.clear_session()

    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(ventana, len(features))
        ),
        tf.keras.layers.LSTM(unidades),
        tf.keras.layers.Dense(1)
    ])

    modelo.compile(
        optimizer="adam",
        loss="mse"
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    )

    historia = modelo.fit(
        X_train_seq,
        y_train_scaled,
        validation_data=(
            X_valid_seq,
            y_valid_scaled
        ),
        epochs=50,
        batch_size=256,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=0
    )

    pred_scaled = modelo.predict(
        X_valid_seq,
        verbose=0
    )

    pred_valid = scaler_y.inverse_transform(
        pred_scaled
    ).ravel()

    pred_valid = np.clip(pred_valid, 0, None)

    resultados_validacion_lstm.append({
        "Configuración": nombre,
        "Ventana": ventana,
        "Unidades LSTM": unidades,
        "Épocas ejecutadas": len(historia.history["loss"]),
        "MAE": mean_absolute_error(
            y_valid_seq,
            pred_valid
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_valid_seq,
                pred_valid
            )
        ),
        "R2": r2_score(
            y_valid_seq,
            pred_valid
        )
    })

    del modelo
    gc.collect()

#mostrar resultados
tabla_validacion_lstm = pd.DataFrame(
    resultados_validacion_lstm
).sort_values("MAE")

print(tabla_validacion_lstm)

#entrenamiento final
train_final = pd.concat(
    [train, validacion],
    ignore_index=True
)

train_final = (
    train_final
    .dropna(subset=columnas_necesarias)
    .sort_values(["comuna", "fecha_hora"])
    .reset_index(drop=True)
)

test_final = (
    test
    .dropna(subset=columnas_necesarias)
    .sort_values(["comuna", "fecha_hora"])
    .reset_index(drop=True)
)

print("Entrenamiento final:", train_final.shape)
print("Test:", test_final.shape)

scaler_X_final = StandardScaler()
scaler_y_final = StandardScaler()

scaler_X_final.fit(train_final[features])
scaler_y_final.fit(train_final[[target]])

ventana = 24

def crear_secuencias_final(df_actual, df_historial=None):

    X_total = []
    y_total = []
    meta_total = []

    for comuna in sorted(df_actual["comuna"].unique()):

        actual = (
            df_actual[df_actual["comuna"] == comuna]
            .sort_values("fecha_hora")
            .reset_index(drop=True)
        )

        if df_historial is None:

            combinado = actual.copy()
            inicio_objetivo = ventana - 1

        else:

            historial = (
                df_historial[df_historial["comuna"] == comuna]
                .sort_values("fecha_hora")
                .tail(ventana - 1)
            )

            combinado = pd.concat(
                [historial, actual],
                ignore_index=True
            )

            inicio_objetivo = 0

        X_escalado = scaler_X_final.transform(
            combinado[features]
        ).astype(np.float32)

        ventanas = sliding_window_view(
            X_escalado,
            window_shape=ventana,
            axis=0
        ).transpose(0, 2, 1)

        y = actual[target].to_numpy(
            dtype=np.float32
        )[inicio_objetivo:]

        metadata = actual[
            ["fecha_hora", "comuna"]
        ].iloc[inicio_objetivo:].copy()

        X_total.append(ventanas)
        y_total.append(y)
        meta_total.append(metadata)

    return (
        np.concatenate(X_total),
        np.concatenate(y_total),
        pd.concat(meta_total, ignore_index=True)
    )

#crear entrenamiento y prueba
X_train_final_seq, y_train_final_seq, _ = (
    crear_secuencias_final(train_final)
)

X_test_final_seq, y_test_final_seq, meta_test_final = (
    crear_secuencias_final(
        test_final,
        df_historial=train_final
    )
)

y_train_final_scaled = scaler_y_final.transform(
    y_train_final_seq.reshape(-1, 1)
).ravel()

print("Train:", X_train_final_seq.shape)
print("Test:", X_test_final_seq.shape)

#entrenar modelo definitivo
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(42)

modelo_lstm_final = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(ventana, len(features))
    ),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1)
])

modelo_lstm_final.compile(
    optimizer="adam",
    loss="mse"
)

modelo_lstm_final.fit(
    X_train_final_seq,
    y_train_final_scaled,
    epochs=23,
    batch_size=256,
    shuffle=False,
    verbose=0
)

print("Modelo LSTM final entrenado.")

#predicción y métricas globales
pred_test_scaled = modelo_lstm_final.predict(
    X_test_final_seq,
    verbose=0
)

pred_test_lstm_original = scaler_y_final.inverse_transform(
    pred_test_scaled
).ravel()

pred_test_lstm = np.clip(
    pred_test_lstm_original,
    0,
    None
)

mae_lstm = mean_absolute_error(
    y_test_final_seq,
    pred_test_lstm
)

rmse_lstm = np.sqrt(
    mean_squared_error(
        y_test_final_seq,
        pred_test_lstm
    )
)

r2_lstm = r2_score(
    y_test_final_seq,
    pred_test_lstm
)

print("LSTM FINAL")
print(f"MAE:  {mae_lstm:.4f} W/m²")
print(f"RMSE: {rmse_lstm:.4f} W/m²")
print(f"R²:   {r2_lstm:.4f}")

#tabla de resultados
resultados_test_lstm = meta_test_final.copy()

resultados_test_lstm["fecha_objetivo"] = (
    resultados_test_lstm["fecha_hora"]
    + pd.Timedelta(hours=1)
)

resultados_test_lstm["ghi_real"] = y_test_final_seq
resultados_test_lstm["ghi_pronosticado"] = pred_test_lstm

resultados_test_lstm["residuo"] = (
    resultados_test_lstm["ghi_real"]
    - resultados_test_lstm["ghi_pronosticado"]
)

print(resultados_test_lstm.head())

#métrica por localidad
metricas_lstm_localidad = []

for comuna, grupo in resultados_test_lstm.groupby("comuna"):

    metricas_lstm_localidad.append({
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

metricas_lstm_localidad = pd.DataFrame(
    metricas_lstm_localidad
)

print(metricas_lstm_localidad)

#residuos
print(
    "Media de residuos:",
    resultados_test_lstm["residuo"].mean()
)

print(
    "Desviación estándar:",
    resultados_test_lstm["residuo"].std()
)

print(
    "Predicciones negativas originales:",
    np.sum(pred_test_lstm_original < 0)
)

#gráficos
import matplotlib.pyplot as plt

for comuna in resultados_test_lstm["comuna"].unique():

    muestra = (
        resultados_test_lstm[
            resultados_test_lstm["comuna"] == comuna
        ]
        .sort_values("fecha_objetivo")
        .head(24 * 7)
    )

    plt.figure(figsize=(14, 5))

    plt.plot(
        muestra["fecha_objetivo"],
        muestra["ghi_real"],
        label="GHI real",
        linewidth=2
    )

    plt.plot(
        muestra["fecha_objetivo"],
        muestra["ghi_pronosticado"],
        label="GHI pronosticado",
        linestyle="--",
        linewidth=2
    )

    plt.title(
        f"Valores reales y pronosticados de GHI en {comuna} — LSTM"
    )

    plt.xlabel("Fecha y hora")
    plt.ylabel("GHI (W/m²)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        f"lstm_real_vs_pronosticado_{comuna}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

#guardar modelo
from google.colab import drive
import os
import joblib
import json

drive.mount("/content/drive")

ruta_modelos = "/content/drive/MyDrive/modelos_radiacion"
os.makedirs(ruta_modelos, exist_ok=True)

modelo_lstm_final.save(
    f"{ruta_modelos}/modelo_lstm_final.keras"
)

joblib.dump(
    scaler_X_final,
    f"{ruta_modelos}/scaler_X_lstm.joblib"
)

joblib.dump(
    scaler_y_final,
    f"{ruta_modelos}/scaler_y_lstm.joblib"
)

configuracion_lstm = {
    "ventana": 24,
    "features": features,
    "target": target,
    "unidades_lstm": 64
}

with open(
    f"{ruta_modelos}/configuracion_lstm.json",
    "w"
) as archivo:
    json.dump(configuracion_lstm, archivo, indent=4)
