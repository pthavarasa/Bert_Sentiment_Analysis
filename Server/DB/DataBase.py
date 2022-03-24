import mysql.connector
import pandas as pd
import pymysql
import sys

class DataBase:
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
			print("New Data Base created")
		except:
			self.cursor.execute(f"USE {self.dbName}")
			print(f"Data Base {self.dbName} already exist -> selected")

	def deleteTable(self, tbName):
		self.cursor.execute(f"DROP TABLE IF EXISTS {tbName}")

	def createTable(self, tbName, tableColumns):
		self.cursor.execute(f"CREATE TABLE {tbName} {tableColumns}")
		self.conn.commit()
		print(f"The table {tbName} has been created successfully")

	def insertElem(self, tbName, columns, values):
		self.cursor.execute(f"INSERT INTO {tbName} {columns} VALUES {values}")
		self.conn.commit()

	def injectDataset(self, tbName, columns, dataset):
		for e in dataset:
			self.insertElem(tbName, columns, e)
		print("The dataset has been inserted successfully")