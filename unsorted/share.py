import argparse
import re
import boto3
from botocore.errorfactory import ClientError
import ast


def main():
    # ------------------------------------- User-friendly command-line interfaces --------------------------------------
    parser = argparse.ArgumentParser(description="Share the file of id 'id' with the user 'user'")
    parser.add_argument("id", type=str, action="store", help="Id of the object you want to share. It should be it's "
                                                             "name. For example '1569618487'")
    parser.add_argument("user", type=str, action="store", help="User to shared with")

    args = parser.parse_args()

    # ---------------------------------------------------- AWS conf ----------------------------------------------------
    # Read the public key, private key and region from the AWS configuration file
    path_to_conf_file = "/opt/rt3d/etc/AWS.conf"
    public_key = ""
    private_key = ""
    region = ""
    with open(path_to_conf_file) as f:
        for line in f:
            result = re.search(r'Public=(.+$)', line, flags=re.MULTILINE)
            if result:
                public_key = result.group(1)
            result = re.search(r'Secret=(.+$)', line, flags=re.MULTILINE)
            if result:
                private_key = result.group(1)
            result = re.search(r'Region=(.+$)', line, flags=re.MULTILINE)
            if result:
                region = result.group(1)

    # --------------------------------------------- Dynamodb communication ---------------------------------------------
    file_id = args.id
    user = args.user
    table = "S3Sync2"
    dynamodb = boto3.client("dynamodb", region_name=region, aws_access_key_id=public_key,
                            aws_secret_access_key=private_key)
    try:
        result = dynamodb.get_item(TableName=table, Key={"Id": {"N": file_id}})
    except ClientError as e:
        print(e.response["Error"]["Message"])
    else:
        # Check if the item exist in our table. If yes then the result of the request should have 'Item' as a key
        if "Item" in result:
            try:
                dict_v = {"SS": [user]}
                attribute_updates = {"Shared with": {"Value": dict_v, "Action": "ADD"}}
                # Dynamo get rid of duplicate value by itself
                dynamodb.update_item(TableName=table, Key={"Id": {"N": file_id}},
                                     AttributeUpdates=attribute_updates)
            except ClientError as e:
                print(e.response["Error"]["Message"])
            else:
                print("The item has been updated")
        else:
            print("Item not found")


if __name__ == "__main__":
    main()
