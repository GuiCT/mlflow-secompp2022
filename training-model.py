# Databricks notebook source
# DBTITLE 1,Importando bibliotecas e dados
# Importando bibliotecas
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
import mlflow
# Importando dados
sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match")
df = sdf.toPandas()

# COMMAND ----------

# DBTITLE 1,Quantidade de memória utilizada
df.info(memory_usage='deep')

# COMMAND ----------

# DBTITLE 1,Separação das colunas
target_column = 'radiant_win'
id_column = 'match_id'

features_columns = list(set(df.columns.tolist()) - set([target_column, id_column]))

y=df[target_column]
X=df[features_columns]

# COMMAND ----------

# DBTITLE 1,Separando treino e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Número de linhas em X_train:", X_train.shape[0])
print("Número de linhas em X_test:", X_test.shape[0])
print("Número de linhas em y_train:", y_train.shape[0])
print("Número de linhas em y_test:", y_test.shape[0])

# COMMAND ----------

# DBTITLE 1,Setando experimento
mlflow.set_experiment("/Users/gc.tomiasi@unesp.br/dota-unesp-gtomiasi")

# COMMAND ----------

# DBTITLE 1,Executando uma run do experimento
with mlflow.start_run():
    mlflow.sklearn.autolog()
    model = ensemble.AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    print(f"Acurácia em treino: {acc_train}")
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    acc_test = metrics.accuracy_score(y_test, y_test_pred)
    print(f"Acurácia em teste: {acc_test}")
