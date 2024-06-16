import duckdb
import pandas as pd
    
import base64 

duckdb.sql("CREATE TABLE images (id VARCHAR, path VARCHAR);")
df=duckdb.sql("SELECT * from read_csv('./data/zna_anotations.csv')").df()
for index, row in df.iterrows():
   # dicom_as_bytes = open("./data/zna_files/"+row['Studienummer']+".dcm", "rb") 
    #with open('data/my.dcm', 'wb') as binary_file:
       #binary_file.write(dicom_as_bytes.read())

 
    #binary=str(dicom_as_bytes.read()).encode("ascii") 
    #base64_bytes = base64.b64encode(binary) 
    #base64_string = base64_bytes.decode("ascii") 
    #SELECT '\xAA\xAB\xAC'::BLOB
    query="INSERT INTO images VALUES ('"+row['Studienummer']+"','data/zna_files/"+row['Studienummer']+".dcm')"
    duckdb.sql(query)
    #dicom_as_bytes.close()

duckdb.sql("COPY (SELECT * FROM images) TO './data/zna_dicom.csv' (HEADER, DELIMITER ',')")

#df=duckdb.sql("SELECT * from read_parquet('./data/zna-images.parquet')").df()
#print(df)
