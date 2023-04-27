# ====== IMPORTS =======================================================================
import sys
import os
from paramiko import SSHClient
from paramiko import RSAKey 

# ====== CONSTANTS =====================================================================

# Acquisition machine login
ACQUISITION_HOST = '169.254.0.1'
ACQUISITION_USER = 'nview'
SYSTEM_KNOWN_KEYS = '/root/.ssh/known_hosts'
SYSTEM_PRIVATE_KEY = '/root/.ssh/id_rsa'

# Directories
WORKSTATION_LOG_FOLDER = "/opt/rt3d/log/"
ACQUISITION_LOG_FOLDER = "/opt/nview-acquisition/log/"

# Strings
WARNING_STRING = "WARNING"
ERROR_STRING = "ERROR"

# ====== FUNCTIONS =====================================================================

def connect_acquisition_client():

    print("Connecting to " + ACQUISITION_HOST)
    try:
        client = SSHClient()
        client.load_system_host_keys(SYSTEM_KNOWN_KEYS)
        private_key = RSAKey.from_private_key_file(SYSTEM_PRIVATE_KEY)
        client.connect(ACQUISITION_HOST, username=ACQUISITION_USER, pkey=private_key)
    except Exception as e:
        print('Failed to connect to Acquisition PC '+  str(e))

    return client

def remote_copy_get(client, target, destination):
    try:
        client = acquisitionClient.open_sftp()
        client.get(target,destination)
        client.close()
    except Exception as e:
        print("Unable to copy file from ssh client: " + target + ", "  + str(e))

def get_warnings_and_errors(log):
    warnings = []
    errors = []
    for line in log:
        if WARNING_STRING in line: warnings.append(line)
        if ERROR_STRING in line: errors.append(line)
    return warnings, errors

def sort_warnings_and_errors(warnings, errors):
    w_dict = {}
    e_dict = {}
    for w in warnings:
        key = w.split(WARNING_STRING, 1)[1]
        if key in w_dict:
            w_dict[key] += 1
        else:
            w_dict[key] = 1
    for e in errors:
        key = e.split(ERROR_STRING, 1)[1]
        if key in e_dict:
            e_dict[key] += 1
        else:
            e_dict[key] = 1
    return w_dict, e_dict

def flatten(l):
    return [item for sublist in l for item in sublist]

def sort_dict(d):
    return dict(sorted(d.items(), key=lambda item: item[1]))

def print_dict_format(d):
    d = sort_dict(d)
    for k,v in d.items():
        print("\t" + str(v) + ": " + k)

def create_directory(path, newfolder):
    outPath = os.path.join(path, newfolder)
    try:
        os.mkdir(outPath)
    except Exception as e:
        print("Unable to create directory: " + outPath + ", " + str(e))
    return outPath 

# ====== MAIN ==========================================================================

def main():

    warnings = []
    errors = []

    # first pull all the local logs
    for file_path in os.listdir(WORKSTATION_LOG_FOLDER):
        if file_path.endswith(".log"):
            f = open(os.path.join(WORKSTATION_LOG_FOLDER,file_path), 'r')
            w, e = get_warnings_and_errors(f)
            warnings.append(w)
            errors.append(e)


    # now the remote logs
    create_directory("/tmp/", "acqLogs")
    acquisition_client = connect_acquisition_client()
    remote_copy_get(acquisition_client,os.path.join(ACQUISITION_LOG_FOLDER,"*"), "/tmp/acqLogs")
    for file_path in os.listdir("/tmp/acqLogs"):
        if file_path.endswith(".log"):
            f = open(os.path.join(WORKSTATION_LOG_FOLDER,file_path), 'r')
            w, e = get_warnings_and_errors(f)
            warnings.append(w)
            errors.append(e)

    # Organize and print the data
    errors = flatten(errors)
    warnings = flatten(warnings)
    w_dict, e_dict = sort_warnings_and_errors(warnings, errors)

    print("Errors:")
    print_dict_format(e_dict)
    print("Warnings:")
    print_dict_format(w_dict)

if __name__ == "__main__":
    main()
