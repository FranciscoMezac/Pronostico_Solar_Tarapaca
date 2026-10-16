# Pronostico_Solar_Tarapaca

RESUMEN:
La radiación solar varía por ciclos diarios y estacionales y por condiciones meteorológicas, lo que
dificulta la estimación de la generación fotovoltaica y complica la integración de fuentes variables
a la red eléctrica. Contar con pronósticos horarios de radiación solar global horizontal (GHI) es,
por tanto, relevante para la planificación energética de la Región de Tarapacá, una de las zonas
con mayor potencial solar del mundo. Este trabajo tuvo como objetivo identificar el enfoque más
adecuado para pronosticar el GHI con una hora de anticipación en tres localidades con
condiciones geográficas contrastantes: Pozo Almonte, Pica y Colchane. Se analizaron 341.856
registros horarios del Explorador Solar del Ministerio de Energía de Chile correspondientes al
periodo 2004-2016. Tras ordenar las series por localidad, se generaron rezagos y variables
temporales cíclicas. El entrenamiento cubrió el periodo 2004-2013, la validación 2014-2015 y la
prueba hasta el 26 de diciembre de 2016. Se evaluaron regresión lineal múltiple, SARIMA, un
perceptrón multicapa (MLP) y una red LSTM mediante las métricas MAE, RMSE y R², truncando a
cero las predicciones negativas. La red LSTM logró el menor MAE (16,24 W/m²) y RMSE (54,67
W/m²), seguida de cerca por el MLP (17,73 W/m² y 55,10 W/m², respectivamente). Colchane, la
localidad altiplánica, presentó los mayores errores en los cuatro enfoques. Las redes neuronales
superaron sistemáticamente a los modelos estadísticos, aunque la diferencia entre LSTM y MLP
fue reducida y su relevancia práctica depende del contexto de implementación. En consecuencia,
se concluye que LSTM es preferible cuando la precisión es prioritaria, mientras que MLP ofrece
un equilibrio favorable entre desempeño y simplicidad operativa para aplicaciones a escala
regional.

