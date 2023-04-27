import pymysql
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import stats

#Database connection variables
MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASSWORD = ''
MYSQL_DATABASE = 'mons'

# ======= FUNCTIONS ========

def insert_into_db(ind, name, hp, a, d, sA, sD, s):
    dbConnection = pymysql.connect(host = MYSQL_HOST, user = MYSQL_USER, password = MYSQL_PASSWORD, database = MYSQL_DATABASE)
    with dbConnection.cursor() as cursor:
        #cursor.execute('select max(ProcedureID) from procedures')
        #procedureID = cursor.fetchone()[0]
        #cursor.execute('select AcquisitionID from proceduresAcquisitionsJunction where ProcedureID = %s', str(procedureID))
        #acqIDs = cursor.fetchall()
        command = 'insert into Stats values(' + str(ind) + "," + "\"" + name + "\"" + "," + str(hp) + "," + str(a) + "," + str(d) + "," + str(sA) + "," + str(sD) + "," + str(s) + ");"
        cursor.execute(command)
        print(command)
        cursor.close()
    dbConnection.commit()

def insert_types_into_db(ind, name, t1, t2):
    dbConnection = pymysql.connect(host = MYSQL_HOST, user = MYSQL_USER, password = MYSQL_PASSWORD, database = MYSQL_DATABASE)
    with dbConnection.cursor() as cursor:
        command = 'insert into types values(' + str(ind) + "," + "\"" + name + "\",\"" + str(t1) + "\",\"" + str(t2) + "\");"
        print(command)
        cursor.execute(command)
        cursor.close()
    dbConnection.commit()

def get_stats_from_json(path, ind):

    ind = ind - 1
    f = open(path)
    data = json.load(f)
    name = data[ind]["name"]["english"]
    hp = data[ind]["base"]["HP"]
    atk = data[ind]["base"]["Attack"]
    defs = data[ind]["base"]["Defense"]
    spa = data[ind]["base"]["Sp. Attack"]
    spd = data[ind]["base"]["Sp. Defense"]
    spe = data[ind]["base"]["Speed"]
    return name, hp, atk, defs, spa, spd, spe

def get_types_from_json(path, ind):

    ind = ind - 1
    f = open(path)
    data = json.load(f)
    name = data[ind]["name"]["english"]
    t1 = data[ind]["type"][0]
    t2 = "NULL"
    if len(data[ind]["type"]) > 1:
        t2 = data[ind]["type"][1]
    return name, t1, t2

def find_strongest_correlation(all_stats):
    max_r = 0
    max_i1 = -1
    max_i2 = -1
    stat_names = ["Attack", "Defense", "Sp. Attack", "Sp. Defense", "Speed"]
    for stat1Index in range(0,len(all_stats[0]) -1 ):
        for stat2Index in range(0, len(all_stats[0]) - 1):
            if stat1Index is stat2Index:
                continue
            s1 = all_stats[:,stat1Index]
            s2 = all_stats[:,stat2Index]
            m, b, r, _, _ = stats.linregress(s1, s2)
            plt.scatter(s1, s2)
            plt.title("r="+str(r))
            plt.xlabel(stat_names[stat1Index])
            plt.ylabel(stat_names[stat2Index])
            best_fit_line = m * np.sort(s1) + b
            plt.plot(np.sort(s1), best_fit_line)
            plt.show()
            if r > max_r:
                max_r = r
                max_i1 = stat1Index
                max_i2 = stat2Index
    return r, max_i1, max_i2


# ========== MAIN ==========
def main():
    all_stats = list()
    for i in range(1, 1 + int(sys.argv[2])):
        name,hp,atk,defs,spa,spd,spe = get_stats_from_json(sys.argv[1], i)
        all_stats.append(np.array([hp,atk,defs,spa,spd,spe]))
        #insert_stats_into_db(i, name, hp, atk, defs, spa, spd, spe)
        name,t1,t2 = get_types_from_json(sys.argv[1], i)
        #insert_types_into_db(i, name, t1,t2)
    
    all_stats = np.array(all_stats)
    print(find_strongest_correlation(all_stats))

if __name__ == "__main__":
    main()
