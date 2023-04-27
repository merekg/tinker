import os
from twilio.rest import Client

account_sid = os.environ['AC36f8e47ca52b30f6e0a73bf090c8a169']
auth_token = os.environ['887d3cb31fc99a233d613382802b9ec8']
myPhoneNumber = "+18015138265"

client = Client(account_sid, auth_token)

message = client.messages.create(body="test", from_=myPhoneNumber, to=myPhoneNumber)
                                 
