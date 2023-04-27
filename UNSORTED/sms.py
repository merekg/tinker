import os
from twilio.rest import Client

account_sid = os.environ['']
auth_token = os.environ['']
myPhoneNumber = ""

client = Client(account_sid, auth_token)

message = client.messages.create(body="test", from_=myPhoneNumber, to=myPhoneNumber)
                                 
