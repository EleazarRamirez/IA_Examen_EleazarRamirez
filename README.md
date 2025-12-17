# IA_Examen_EleazarRamirez
practica de IA que permita clasificar o predecir un comportamiento.
## 2.1 Objetivo del proyecto
se realizo la practica con el fin de construir un modelo de machine learning que permita predecir si un pasajero del titatic sobrevivio o no (1 significa que si sobrevivio, y 0 significa que no sobrevivio. 

## 2.2 modelo utilizado
utilizamos el modelo ARBOL DE DESICIÓN, porque nos permite visualizar claramente las reglas de clasificacion.
## 2.3 descripcion del dataset 
este contiene informacion real de los pasajeros. lo que hice con los datos fue: limpieza de nulos. con esto se imputaron los datos faltantes de la columna edad(age), utilizando la mediana obtenida con los otros datos que si estan. y tambien converti los datos de 'sex' y 'embarked' en numero para que puedan ser evaluados. ya que machine learning no reconoce texto directo para procesarlo.
## 2.4 instrucciones para ejecutar el proyecto
1. clonar el repositorio
2. instalar librerias: " pip install pandas numpy scikit-learn matplotlib seaborn"
nota: seaborn yo lo use para descargar el archvio csv original del titatic y asi cargarlo a mi app.py
3. ejecutar el script principal: 'python app.py'

preguntas:
**¿que mejoras se presentaron? la integracion de las edades faltantes. con esto se mejoro la estadistica de los datos que teniamos.
**¿que mejoras se podian aplicar? creo que se pudo haber realizado la extraccion de titutlos para saber si era señor o señora(mr, mrs) con esto se pudo haber mejorado la prediccion de supervivencia.
**justificacion del modelo: elegi el arbol de desicion porque , en contexto de supervivencia del titatic, las desiciones suelen ser jerarquicas como en la pelicula(fueron mujeres y niños primero), lo cual el arbol lo entenderia perfectamente.
