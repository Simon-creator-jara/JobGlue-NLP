import sys
from awsglue.dynamicframe import DynamicFrame
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, udf
from pyspark.sql.functions import lower
from pyspark import SparkFiles 
import pandas as pd
import json
from pandas import json_normalize
from numpy import NaN
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import googletrans
from googletrans import Translator
from urllib.request import urlopen
import urllib.request
import requests 
import warnings
import boto3
from pyspark.sql.types import *
from pyspark.sql import SQLContext
warnings.filterwarnings('ignore')
import regex as re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words(['spanish','english'])
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import es_core_news_sm
nlp = es_core_news_sm.load()
import joblib
import numpy as np
from datetime import datetime
from datetime import date

args = getResolvedOptions(sys.argv,
                            ['JOB_NAME',
                            'key',
                            'bucket_origin',
                            'bucket_destiny'])
database="modeloentrenado"
table = "maintable"
ruta_archivo_json = 's3://' + args['bucket_origin'] + '/' + args['key']
target_s3_path = args['bucket_destiny'] 
sc = SparkContext()
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)


datos=pd.read_json(ruta_archivo_json)
datos=datos.drop(columns=['locationInfos','details','header','title','activityControls','products'])
datos=datos.dropna()
datos=datos.iloc[2000:2010,:]
sparkDF=spark.createDataFrame(datos) 


###Funciones

def fechas(string):
  fecha=datetime.strptime((string[0:10]).replace('-','/'), "%Y/%m/%d")
  return fecha

def hora(string):
  hora=datetime.strptime((string[0:10]+' '+string[11:19]).replace('-','/'), "%Y/%m/%d %H:%M:%S")
  return hora

def hora_def(string):
  hora=string.time()
  return hora

def dia_def(string):
  hora=string.date()
  return hora

def LimpiarCadena(string_url):
  url_def=''
  if '&usg' in string_url.lower():
    pos=string_url.index('&usg')
    url_def=string_url[29:pos]
  else:
    url_def=string_url
  
  return url_def

def metaTag(url_inv):
    try:
        hdr = {'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Mobile Safari/537.36'}

        req = requests.get(url_inv,headers=hdr, timeout=30)
        content = req.content
        soup = BeautifulSoup(content, features="html.parser")
        
    
        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()   # rip it out
    
    
        # get text
        text = soup.get_text()
        
        if len(text)<=60000:
    
            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
        
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = ' '.join(chunk for chunk in chunks if chunk)
        else:
            text=text[0:10000]
            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
        
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = ' '.join(chunk for chunk in chunks if chunk)
            

    except ConnectionError:
        
        text=NaN
    except TimeoutError:
        text=NaN
    
    except ConnectionError:
        text=NaN

    except:

        text=NaN

    try:
        print(url_inv,'\n',text,'\n',len(text))
    except:
        print(url_inv,'\n',text)
    

    return text

fechasUDF = udf(lambda x:fechas(x),DateType()) 
hora_defUDF = udf(lambda x:hora(x),	TimestampType()) 
LimpiarCadenaUDF = udf(lambda x:LimpiarCadena(x),StringType()) 
metaTagUDF = udf(lambda x:metaTag(x),StringType()) 

sparkDF=sparkDF.withColumn("Fecha_Consulta", fechasUDF(col("time")))

sparkDF=sparkDF.withColumn("urlLimpia", LimpiarCadenaUDF(col("titleUrl")))
sparkDF=sparkDF.withColumn("Descripcion", metaTagUDF(col("urlLimpia")))

sparkDF=sparkDF.na.drop()
sparkDF=sparkDF.drop('titleUrl','products')


patternURLEMAIL=r'((http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/|www\.))'
patterncom=r'(.com)'
patternHashtagMention=r'(@\w+)|(#\w+)'
#Primero convertimos a minúscula
sparkDF = sparkDF.withColumn('Descripcion', lower(col('Descripcion')))
# Utilizamos las expresiones regulares anteriores sobre URL, email, hashtag y menciones para quitarlos
udfemail=udf(lambda elem: re.sub(patternURLEMAIL,' ', elem),StringType())
udfcom=udf(lambda elem: re.sub(patterncom,' ', elem),StringType())
sparkDF = sparkDF.withColumn('Descripcion', udfemail(col('Descripcion')))
sparkDF = sparkDF.withColumn('Descripcion', udfcom(col('Descripcion')))
# Utilizamos una expresión regular para eliminar los hashtag y las menciones con @
udfhashtag=udf(lambda elem: re.sub(patternHashtagMention,' ', elem),StringType())
sparkDF = sparkDF.withColumn('Descripcion', udfhashtag(col('Descripcion')))
# Utilizamos una expresión regular también para eliminar los signos de mayor que y menor que
udfsignos=udf(lambda elem: re.sub(r'(\&gt\;)|(\&lt\;)',' ', elem),StringType())
sparkDF = sparkDF.withColumn('Descripcion', udfsignos(col('Descripcion')))
# Utilizamos una expresión regular también para eliminar a.m y p.m cuando mencionan horas
udfampm=udf(lambda elem: re.sub(r'(a\.m)|(p\.m)',' ', elem),StringType())
sparkDF = sparkDF.withColumn('Descripcion', udfampm(col('Descripcion')))
# Utilizamos una expresión regular también para eliminar los números
udfnum=udf(lambda elem: re.sub(r'\d+',' ', elem),StringType())
sparkDF = sparkDF.withColumn('Descripcion',udfnum(col('Descripcion')))
udfpipe=udf(lambda elem: re.sub(r'[|]',' ', elem),StringType())
sparkDF = sparkDF.withColumn('Descripcion', udfpipe(col('Descripcion')))
udfpunto=udf(lambda elem: re.sub(r'[.]',' ', elem),StringType())
sparkDF = sparkDF.withColumn('Descripcion', udfpunto(col('Descripcion')))
udfcoma=udf(lambda elem: re.sub(r'[,]',' ', elem),StringType())
sparkDF = sparkDF.withColumn('Descripcion', udfcoma(col('Descripcion')))
udfmayus=udf(lambda elem: re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚ]',' ', elem),StringType())
sparkDF = sparkDF.withColumn('Descripcion', udfmayus(col('Descripcion')))
## Sustituir espacios de más
udfespaci=udf(lambda elem: re.sub(r'\s+',' ', elem),StringType())
sparkDF = sparkDF.withColumn('Descripcion', udfespaci(col('Descripcion')))
## Eliminar signos de puntuación '[!#?,.:";]'
#df['Descripcion'] = df['Descripcion'].apply(lambda elem: re.sub(r"""[‘’]""",' ', elem))
non_words = ['!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~']
non_words.extend(['¿', '¡', '‘', '’','.',','])
udfnon=udf(lambda elem: ''.join([c for c in elem if c not in non_words]),StringType())
sparkDF = sparkDF.withColumn('Descripcion', udfnon(col('Descripcion')))

udfletters=udf(lambda x: ' '.join([word for word in x.split() if len(word)>2 and len(word)<15]),StringType())
sparkDF = sparkDF.withColumn('Descripcion', udfletters(col('Descripcion')))

stopudf = udf(lambda x: ' '.join([word for word in x.split() if word not in (stop)]),StringType())
sparkDF = sparkDF.withColumn('Descripcion_traducida', stopudf(col('Descripcion'))) 

pandasDF = sparkDF.toPandas()

pandasDF['Descripcion_traducida'] = pandasDF['Descripcion_traducida'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
pandasDF['tokens'] = pandasDF['Descripcion_traducida'].apply(lambda x: word_tokenize(x))
pandasDF['tokens_clean']=pandasDF['tokens'].apply(lambda text: [word for word in text if word.isalnum() and len(word)>1])
pandasDF['lemmas'] = pandasDF.Descripcion_traducida.apply(lambda text: [tok.lemma_ for tok in nlp(text)])

df_example=pd.read_csv('s3://proeycto-bigdata2023/lista_interes_def.csv')
sc.addFile ("s3://proeycto-bigdata2023/modelo_entrenado_def.pkl")
ab = joblib.load(SparkFiles.get("modelo_entrenado_def.pkl"))

lista_verificar=list(df_example['0'])
def verificar_lemmas(lista,lista_verificar):
  lista_def=[]
  for i in range(len(lista)):
    if lista[i] in lista_verificar:
      lista_def.append(lista[i])
  return lista_def

pandasDF['verificacion_lemmas']=pandasDF['lemmas'].apply(lambda x: verificar_lemmas(x,lista_verificar))

vectorizacion=[]

for i in range(len(lista_verificar)):  
  lista=[]
  for j in range(len(pandasDF.verificacion_lemmas)):
    lista.append(pandasDF.verificacion_lemmas[j].count(lista_verificar[i]))
  vectorizacion.append(lista)

pred=np.array(vectorizacion)
pred=pred.T
prediccion=ab.predict(pred)
pandasDF['prediccion'] = pd.DataFrame(prediccion, index= pandasDF.index)
print(pandasDF)

sparkDF=spark.createDataFrame(pandasDF) 
client = boto3.client('glue')

try :
  client.get_database(Name= database)
  client.get_table(DatabaseName=database,Name=table)
  table_exists = True
except Exception:
  table_exists=False

if table_exists:
  sparkDF.write.mode("append").options(header='True', delimiter=',').option('path',target_s3_path+"maintable").format('parquet').partitionBy("Fecha_Consulta").saveAsTable(f"{database}.{table}")
else:
  sparkDF.write.options(header='True', delimiter=',').option('path',target_s3_path+"maintable").format('parquet').partitionBy("Fecha_Consulta").saveAsTable(f"{database}.{table}")

##############Productividad####################################

from datetime import datetime
from datetime import timedelta
from datetime import time 

pandasDF['Hora_Consulta']=pandasDF['time'].apply(hora)
pandasDF['Fecha_Consulta']=pandasDF['Hora_Consulta'].apply(dia_def)
pandasDF['Hora_definitiva_Consulta']=pandasDF['Hora_Consulta'].apply(hora_def)
lista=[]
lista.append(0)
for i in range(len(pandasDF)):
  
  if i < len(pandasDF)-1:
    if pandasDF['Fecha_Consulta'][i] == pandasDF['Fecha_Consulta'][i+1]:
      
      lista.append(timedelta(seconds=pandasDF['Hora_definitiva_Consulta'][i].second,hours=pandasDF['Hora_definitiva_Consulta'][i].hour,minutes=pandasDF['Hora_definitiva_Consulta'][i].minute)-timedelta(seconds=pandasDF['Hora_definitiva_Consulta'][i+1].second,hours=pandasDF['Hora_definitiva_Consulta'][i+1].hour,minutes=pandasDF['Hora_definitiva_Consulta'][i+1].minute))
    else:
      lista.append(0)

pandasDF['diferencia_tiempo']=pd.DataFrame(lista,index=pandasDF.index)

def convertir_fecha(string):
  return pd.Timestamp(string)

def convertir_sec(string):
  try:
    return string.seconds
  except:
    return string

pandasDF['Fecha_Consulta']=pandasDF['Fecha_Consulta'].apply(convertir_fecha)
pandasDF['diferencia_tiempo']=pandasDF['diferencia_tiempo'].apply(convertir_sec)

def creacion_prod_noprod(df,columna_prod,columna_tiempo,inicio_office,fin_office,columna_hora):
  lista_prod_office=[]
  lista_noprod_office=[]
  lista_prod=[]
  lista_noprod=[]
  for i in range(len(df)):
    if df[columna_prod][i]==1 and df[columna_hora][i]>=inicio_office and df[columna_hora][i]<=fin_office:
      lista_prod_office.append(df[columna_tiempo][i])
      lista_noprod_office.append(0)
      lista_prod.append(0)
      lista_noprod.append(0) 
    elif df[columna_prod][i]==0 and df[columna_hora][i]>=inicio_office and df[columna_hora][i]<=fin_office:
      lista_prod_office.append(0)
      lista_noprod_office.append(df[columna_tiempo][i])  
      lista_prod.append(0)
      lista_noprod.append(0)     
    elif df[columna_prod][i]==1:
      lista_prod.append(df[columna_tiempo][i])
      lista_noprod.append(0)  
      lista_prod_office.append(0)
      lista_noprod_office.append(0)     
    else:
      lista_prod.append(0)
      lista_noprod.append(df[columna_tiempo][i])
      lista_prod_office.append(0)
      lista_noprod_office.append(0)   

  df['Tiempo_productivo_oficina']=pd.DataFrame(lista_prod_office,index=df.index)
  df['Tiempo_No_productivo_oficina']=pd.DataFrame(lista_noprod_office,index=df.index)
  df['Tiempo_productivo_fuera_oficina']=pd.DataFrame(lista_prod,index=df.index)
  df['Tiempo_No_productivo_fuera_oficina']=pd.DataFrame(lista_noprod,index=df.index)
  return df

pandasDF=creacion_prod_noprod(pandasDF,'prediccion','diferencia_tiempo',time(7, 0, 0),time(18, 0, 0),'Hora_definitiva_Consulta')

respuesta=pandasDF.groupby(pd.Grouper(key='Fecha_Consulta', axis=0, 
                      freq='1D', sort=True)).sum()

def porcentaje_prod(df,tiempoProdofi,tiempoProd,total):
  lista_ofi=[]
  lista_no=[]
  for i in range(len(df)):
    lista_ofi.append(df[tiempoProdofi][i]/df[total][i])
    lista_no.append(df[tiempoProd][i]/df[total][i])

  
  df['Porcentaje_Prod_Ofi']=pd.DataFrame(lista_ofi,index=df.index)
  df['Porcentaje_Prod_Fuera_Ofi']=pd.DataFrame(lista_no,index=df.index)

  return df

respuesta=porcentaje_prod(respuesta,'Tiempo_productivo_oficina','Tiempo_productivo_fuera_oficina','diferencia_tiempo')

respuesta['Porcentaje_Prod_Ofi'].fillna(value=0,inplace=True)
respuesta['Porcentaje_Prod_Fuera_Ofi'].fillna(value=0,inplace=True)
respuesta['Fecha_Consulta']=respuesta.index
sparkrespuesta=spark.createDataFrame(respuesta)

database2="productivity"
table2="decision"

try :
  client.get_database(Name= database2)
  client.get_table(DatabaseName=database2,Name=table2)
  table_exists = True
except Exception:
  table_exists=False

if table_exists:
  sparkrespuesta.write.mode("append").options(header='True', delimiter=',').option('path',target_s3_path+"productivity").format('parquet').saveAsTable(f"{database2}.{table2}")
else:
  sparkrespuesta.write.options(header='True', delimiter=',').option('path',target_s3_path+"productivity").format('parquet').saveAsTable(f"{database2}.{table2}")
job.commit()