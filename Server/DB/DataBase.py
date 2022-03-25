import mysql.connector
import pandas as pd
import sys
import pymysql

class Database:
	"""
	Class de la base de données
	"""

	def __init__(self, host, user, password, dbName, tbName):
		self.dbName = dbName
		self.tbName = tbName
		self.cursor = None
		self.connexion(host, user, password)
		self.createOrSelect()
		self.deleteTable(tbName)

	def connexion(self, host, user, password):
		try:
			self.conn = mysql.connector.connect(
				host = host,
				user = user,
				password = password,
			)
			self.cursor = self.conn.cursor()
			print(f"Connected to DB:{host}")

		except Exception as e:
			print(f"Error in connexion:{str(e)}")
			sys.exit(1)

	def createOrSelect(self):
		try:
			self.cursor.execute(f"CREATE DATABASE {self.dbName}")
			self.conn.commit()
			print("New DataBase created")
		except:
			self.cursor.execute(f"USE {self.dbName}")
			print(f"DataBase {self.dbName} already exist -> selected")

	def insertElem(self, tbName, columns, values):
		self.cursor.execute(f"INSERT INTO {tbName} {columns}VALUES{values}")
		self.conn.commit()
		print("The element has been inserted Successfully")
	## dataBase.insertElem(conn, dbCursor, "data", col, values)

	def deleteTable(self, tbName):
		self.cursor.execute(f"DROP TABLE IF EXISTS {tbName}")

	def createTable(self, tbName, tbColumns):
		self.cursor.execute(f"CREATE TABLE {tbName} {tbColumns}")
		self.conn.commit()
		print(f"The table {tbName} has been created successfully")

	def injectFile(self, path, tbName):
		self.cursor.execute(f"LOAD DATA LOCAL INFILE '{path}' INTO TABLE {tbName} FIELDS TERMINATED BY ';' ENCLOSED BY '\n' IGNORE 1 LINES;")
		self.conn.commit()
		print(f"Successfully loaded {path} into {tbName}")

	def getDfOfDataset(self, tbName):
		conn = pymysql.connect(host='localhost',
                        user='root',
                        password='')
		df = pd.read_sql(f"Select * From {self.dbName}.{tbName}", conn)
		df.dropna(inplace=True)
		df.reset_index(drop=True, inplace=True)
		conn.close()
		return df
