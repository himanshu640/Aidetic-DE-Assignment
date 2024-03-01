# Databricks notebook source
# MAGIC %md
# MAGIC Load the  dataset into a PySpark DataFrame.

# COMMAND ----------

DF = spark.read.format("csv").option("header",True).option("inferschema",True).load("/FileStore/tables/database-1.csv")

DF.display()

# COMMAND ----------

# DBTITLE 1,Convert the Date and Time columns into a timestamp column named Timestamp.
Df_new=DF.selectExpr("*","to_timestamp(Date,'M/d/yyyy') as timestamp","to_timestamp(Time,'M/d/yyyy') as timestamp_two")

Df_new.display()

# COMMAND ----------

# MAGIC %md
# MAGIC .Filter the dataset to include only earthquakes with a magnitude greater than 5.0.

# COMMAND ----------

DF_magnitude=Df_new.where(Df_new["Magnitude"] > 5.0)
DF_magnitude.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Calculate the average depth and magnitude of earthquakes for each earthquake type.

# COMMAND ----------

from pyspark.sql import functions

DF_latest=DF_magnitude.groupBy('Type').agg(functions.avg('Depth').alias("average_depth"),
                                           functions.avg('Magnitude').alias("average_magnitude"))

DF_latest.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Implement a UDF to categorize the earthquakes into levels (e.g., Low, Moderate, High) based on their magnitudes.

# COMMAND ----------

def mag_level(magnitude):
    if magnitude <= 5.5:
        return "Low"
    elif magnitude > 5.6 and magnitude <7.0:
        return "Moderate"
    else:
        return "High"

# COMMAND ----------

from pyspark.sql.types import StringType

spark.udf.register("earthquake_levels",f=mag_level,returnType=StringType())

# COMMAND ----------

Df_new.createOrReplaceTempView('earthquake')

# COMMAND ----------

Result = spark.sql("""
    SELECT *,earthquake_levels(magnitude) AS Magnitude_level from earthquake
    """)

Result.display()

# COMMAND ----------

# MAGIC %md
# MAGIC .Calculate the distance of each earthquake from a reference location (e.g., (0, 0)).

# COMMAND ----------

from pyspark.sql.functions import col,acos,sin,cos,sin,lit,radians,toRadians,sqrt

refrence_lat = 0
refrence_lon = 0

R = 6371.0

df_with_radians=Result.withColumn('lat_radian',radians(Df_new['Latitude']))\
    .withColumn('lon_radian',radians(Df_new['Longitude']))\
        .withColumn('ref_lat_rad', radians(lit(refrence_lat)))\
            .withColumn('ref_loan_rad', radians(lit(refrence_lon)))

dlat = df_with_radians["lat_radian"] - df_with_radians["ref_lat_rad"]
dlon = df_with_radians["lon_radian"] - df_with_radians["ref_loan_rad"]

a = (sin(dlat / 2) ** 2) + (cos(df_with_radians["lat_radian"]) * cos(df_with_radians["ref_lat_rad"]) * sin(dlon / 2) ** 2)
c = 2 * acos(sqrt(a))

distance_km = R * c

DF_distance = df_with_radians.withColumn("Distance_From_Reference", distance_km)

DF_distance=DF_distance.selectExpr("*","Type as Types")

# COMMAND ----------

DF_join=DF_distance.join(DF_latest,DF_distance.Types == DF_latest.Type,"inner")

DF_join.display()

# COMMAND ----------

Final=DF_join.select('timestamp','Latitude','Longitude','Types','Depth','Magnitude','Magnitude_level','average_depth','average_magnitude','Distance_From_Reference')

# COMMAND ----------

Final.write.csv("/FileStore/tables/final1.csv",header=True)

# COMMAND ----------

pip install folium

# COMMAND ----------

# MAGIC %md
# MAGIC .Visualize the geographical distribution of earthquakes on a world map using appropriate libraries (e.g., Basemap or Folium).

# COMMAND ----------

import folium



earthquake_locations = spark.sql("SELECT Latitude, Longitude FROM earthquake").collect()


map_center = [0, 0]
map_zoom = 2 
world_map = folium.Map(location=map_center, zoom_start=map_zoom)


for row in earthquake_locations:
    folium.Marker(location=[row.Latitude, row.Longitude], popup="Earthquake").add_to(world_map)

world_map.save("earthquake_distribution_map.html")
world_map


# COMMAND ----------


