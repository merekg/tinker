import json

import subprocess

output = subprocess.check_output("aws ec2 describe-instances", shell=True)
output =json.loads(output)
for res in output['Reservations']:
	for instances in res['Instances']:
		for tag in instances['Tags']:
			if tag['Key'] == 'Name':
				print tag['Key'],
				print ' '+tag['Value'],	
				print 'InstanceId:' + str(instances.get('InstanceId','')),
				print 'PublicIP:' + str(instances.get('PublicIpAddress',''))
