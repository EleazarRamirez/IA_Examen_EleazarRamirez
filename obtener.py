import seaborn as sns

# Cargar el dataset que viene integrado en la librería de seaborn
df = sns.load_dataset('titanic')

# Guardar el archivo csv
df.to_csv('titanic.csv', index=False)