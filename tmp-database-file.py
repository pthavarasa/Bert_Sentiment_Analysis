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
			print(f"Connected to DB:{host}")
			return connexion, connexion.cursor()

		except Exception as e:
			print(f"Error in connexion:{str(e)}")
			sys.exit(1)

	def createDB(self, db, cursor):
		cursor.execute(f"CREATE DATABASE {self.DB_Name}")
		db.commit()
		print("New Data Base created")

	def selectDB(self, cursor):
		cursor.execute(f"USE {self.DB_Name}")
		print(f"Data Base {self.DB_Name} already exist ; selected")

	def createTable(self,db, cursor, tbName, tableColumns):
		cursor.execute(f"CREATE TABLE {tbName}{tableColumns}")
		db.commit()
		print(f"The table {tbName} has been created Successfully")

	# Ne fonctionne pas -> A supprimer ?
	def selectTable(self, cursor, tbName):
		cursor.execute(f"USE TABLE {tbName}")

	def showDb(self, cursor):
		cursor.execute("SHOW DATABASES")
		for db in cursor:
			print(db)

	def showTables(self, cursor):
		cursor.execute("SHOW TABLES")
		for tb in cursor:
			print(tb)

	def insertElem(self, db, cursor, tbName, columns, values):
		cursor.execute(f"INSERT INTO {tbName} {columns}VALUES{values}")
		db.commit()
		print("The element has been inserted Successfully")
	## dataBase.insertElem(conn, dbCursor, "data", col, values)

	def deleteElem(self, db, cursor, tbName, condition):
		cursor.execute(f"DELETE FROM {tbName} WHERE {condition}")
		db.commit()
		print("The element has been deleted Successfully")

	def deleteTable(self, cursor, tbName):
		cursor.execute(f"DROP TABLE IF EXISTS {tbName}")
		print(f"The table {tbName} has been deleted Successfully")

	def selectColumn(self, cursor, columns, tbName):
		cursor.execute(f"SELECT {columns} from {tbName}")
		res = cursor.fetchall()
		print("\nThe columns {} selected from {} are :".format(columns,tbName))
		for line in res:
			print(line)

	def selectElems(self, cursor, columns, tbName, condition):
		cursor.execute(f"SELECT {columns} from {tbName} WHERE {condition}")
		res = cursor.fetchall()
		print("\nThe elements whith the condition {} are :".format(condition))
		for line in res:
			print(line)

	def selectOneElem(self, cursor, columns, tbName, condition):
		cursor.execute(f"SELECT {columns} from {tbName} WHERE {condition}")
		res = cursor.fetchone()
		print("\nThe element whith the condition {} is :".format(condition))
		for line in res:
			print(line)

	def selectAll(self, cursor, tbName):
		cursor.execute(f"SELECT * from {tbName}")
		res = cursor.fetchall()
		print("\nThe Select All of the table {} :".format(tbName))
		for line in res:
			print(line)

	def getAll(self, cursor, tbName):
        cursor.execute(f"SELECT * from {tbName}")
        return cursor.fetchall()

	def updateElem(self, db, cursor, tbName, newValue, value):
		cursor.execute(f"UPDATE {tbName} SET {newValue} WHERE {value}")
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

"""

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

"""