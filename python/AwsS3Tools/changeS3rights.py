import sys
import subprocess
import string
import json
import os
import time


# Processing function
def subprocess_cmd(command):
    #print (command)
    process = subprocess.Popen(command,stdout=subprocess.PIPE,shell=True)
    return process
    #proc_stdout = process.communicate()[0].strip()
    #print proc_stdout

if(len(sys.argv) > 1):
	nextToken = str(sys.argv[1])

print "Starting changeS3rights with token " + nextToken

# Parameters
nextToken = 'eyJDb250aW51YXRpb25Ub2tlbiI6IG51bGwsICJib3RvX3RydW5jYXRlX2Ftb3VudCI6IDV9'
processes = 6
batchSize = 1000 #1000 max or command buffer may overflow
batchesProcessed = 0

# Process batches
while(True):
	# List objects in current batch
	listingProc = subprocess.Popen(['aws', 's3api', 'list-objects-v2', '--bucket', 'companydatasets', '--starting-token', nextToken, '--max-items', str(batchSize), '--prefix', 'CT_scans/', '--query', '[Contents[].{Key: Key}, NextToken]'],stdout=subprocess.PIPE)
	std_out, std_error = listingProc.communicate()
	data = json.loads(std_out)
	keys = data[0]
	# print keys[0]['Key'] test on how to use keys
	nextToken = data[1]
	
	# Create command srting for each process
	commands = []
	for p in range (0, processes):
	   commands.append('')
	   for k in range (0, batchSize):
		if (k % processes == p):
		   subcommand ='aws s3api put-object-acl --bucket companydatasets --key {} --grant-full-control emailaddress=cristian.atria@companymed.com --profile actualmed;'.format(keys[k]['Key'])
		   commands[p] += subcommand
	
	# Distribute processing
	listOfProcesses = []	
	for p in range (0, processes):
		listOfProcesses.append(subprocess_cmd(commands[p]))

	# Communicate status when done
	for p in range (0, processes):
		listOfProcesses[p].communicate()	# Be sure to be done, rejoin the processes
	batchesProcessed += 1
	objectsProcessed = batchesProcessed * batchSize 
	print str(objectsProcessed) + " objects processed, next token: " +str(nextToken)



