import sqlite3

connection = sqlite3.connect("statistic.db")
cursor = connection.cursor()
res = cursor.execute("select * from statistic")

cols = res.description
stats = tuple(res)[0]
for i in range(3):
    print(cols[i][0], ":", stats[i])

connection.close()
