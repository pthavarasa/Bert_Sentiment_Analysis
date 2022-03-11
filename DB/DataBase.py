import mysql.connector
import pandas as pd
import pymysql
import sys

class DataBase:
	"""
	Class de la base de données
	"""

	def __init__(self, DB_Name):
		self.DB_Name = DB_Name

	def connexionDB(self, host, user, password):
		try:
			connexion = mysql.connector.connect(
				host = host,
				user = user,
				password = password,
			)
			print("Connected to DB:{}".format(host))
			return connexion, connexion.cursor()

		except Exception as e:
			print("Error in connexion:{}".format(str(e)))
			sys.exit(1)

	def createDB(self, db, cursor):
		cursor.execute("CREATE DATABASE {}".format(self.DB_Name))
		db.commit()
		print("New Data Base created")

	def selectDB(self, cursor):
		cursor.execute("USE {}".format(self.DB_Name))
		print("Data Base {} already exist ; selected".format(self.DB_Name))

	def createTable(self,db, cursor, tbName, tableColumns):
		cursor.execute("CREATE TABLE {}{}".format(tbName,tableColumns))
		db.commit()
		print("The table {} has been created Successfully".format(tbName))

	def selectTable(self, cursor, tbName):#############marche pas
		cursor.execute("USE TABLE {}".format(tbName))

	def showDb(self, cursor):
		cursor.execute("SHOW DATABASES")
		for db in cursor:
			print(db)

	def showTables(self, cursor):
		cursor.execute("SHOW TABLES")
		for tb in cursor:
			print(tb)

	def insertElem(self, db, cursor, tbName, columns, values):
		cursor.execute("INSERT INTO {} {}VALUES{}".format(tbName, columns, values))
		db.commit()
		print("The element has been inserted Successfully")

	def deleteElem(self, db, cursor, tbName, condition):
		cursor.execute("DELETE FROM {} WHERE {}".format(tbName, condition))
		db.commit()
		print("The element has been deleted Successfully")

	def deleteTable(self, cursor, tbName):
		cursor.execute("DROP TABLE IF EXISTS {}".format(tbName))
		print("The table {} has been deleted Successfully".format(tbName))

	def selectColumn(self, cursor, columns, tbName):
		cursor.execute("SELECT {} from {}".format(columns, tbName))
		res = cursor.fetchall()
		print("\nThe columns {} selected from {} are :".format(columns,tbName))
		for line in res:
			print(line)

	def selectElems(self, cursor, columns, tbName, condition):
		cursor.execute("SELECT {} from {} WHERE {}".format(columns, tbName, condition))
		res = cursor.fetchall()
		print("\nThe elements whith the condition {} are :".format(condition))
		for line in res:
			print(line)

	def selectOneElem(self, cursor, columns, tbName, condition):
		cursor.execute("SELECT {} from {} WHERE {}".format(columns, tbName, condition))
		res = cursor.fetchone()
		print("\nThe element whith the condition {} is :".format(condition))
		for line in res:
			print(line)

	def selectAll(self, cursor, tbName):
		cursor.execute("SELECT * from {}".format(tbName))
		res = cursor.fetchall()
		print("\nThe Select All of the table {} :".format(tbName))
		for line in res:
			print(line)

	def updateElem(self, db, cursor, tbName, newValue, value):
		cursor.execute("UPDATE {} SET {} WHERE {}".format(tbName, newValue, value))
		db.commit()
		print("The element has been updated Successfully")

	def updateBD(self, db):
		db.commit()

	def loadFile(self, db, cursor, path, tbName):
		cursor.execute("LOAD DATA LOCAL INFILE {} INTO TABLE {} FIELDS TERMINATED BY ';' ENCLOSED BY '\n' IGNORE 1 LINES;".format(path,tbName))
		db.commit()
		print("\nSuccessfully loaded the table from csv")

	def closeDB(self, cursor, connexion):
		connexion.close()
		cursor.close()



def main():
	dataBase = DataBase("PROJET")
	conn, dbCursor = dataBase.connexionDB("localhost","root","")
	dataBase.showDb(dbCursor)

	try:
		dataBase.createDB(conn, dbCursor)
	except:
		dataBase.selectDB(dbCursor)
		dataBase.showTables(dbCursor)

	columns = "(name VARCHAR(255), sentiment VARCHAR(255), review VARCHAR(255))"

	try:
		dataBase.createTable(conn, dbCursor,"data",columns)
	except:
		dataBase.selectAll(dbCursor, "data")

	
	col = "(name, sentiment, review)"
	column = "name, review"
	condition = "name = 'ryry'"
	values = ("radja", "a", "b")

	dataBase.insertElem(conn, dbCursor, "data", col, values)
	dataBase.selectAll(dbCursor,"data")
	dataBase.selectColumn(dbCursor,column,"data")
	dataBase.selectElems(dbCursor,column,"data",condition)
	dataBase.updateElem(conn, dbCursor, "data", "name='ryry'","name ='radja'")

	path = """'C:/Users/ThinkPad/Desktop/NewProjet/datasetTest.txt'"""
	tableName = "data"
	dataBase.loadFile(conn, dbCursor, path, tableName)

	dataBase.closeDB(dbCursor, conn)

if __name__ == "__main__":
    main()
