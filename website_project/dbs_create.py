import sqlite3

connection = sqlite3.connect("statistic.db")
cursor = connection.cursor()
cursor.execute("create table statistic (Original integer, Low integer, Medium integer, High integer )")


cursor.execute("insert into statistic values (0,0,0,0)")

#for row in cursor.execute("select * from statistic"):
#    print(row)

connection.commit()
connection.close()
