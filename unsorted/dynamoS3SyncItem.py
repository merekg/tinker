import argparse
import re
import boto3
from botocore.errorfactory import ClientError


def main():
    # ------------------------------------- User-friendly command-line interfaces --------------------------------------
    parser = argparse.ArgumentParser(description="Add or modify the metadata of an object store in S3 into dynamo")
    parser.add_argument("bucket", type=str, action="store", help="The S3 bucket where the object is stored")
    parser.add_argument("path", type=str, action="store", help="The path of the object within the bucket")

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

    # ------------------------------------------------ S3 communication ------------------------------------------------
    bucket = args.bucket
    path = args.path

    # Get the object metadata specified by the user
    s3 = boto3.client("s3", region_name=region, aws_access_key_id=public_key, aws_secret_access_key=private_key)
    try:
        s3_obj_metadata = s3.head_object(Bucket=bucket, Key=path)
    except ClientError as e:
        print(e.response["Error"]["Message"])
    else:
        # ------------------------------------------- Dynamodb communication -------------------------------------------
        table = "S3Sync2"
        dynamodb = boto3.client("dynamodb", region_name=region, aws_access_key_id=public_key,
                                aws_secret_access_key=private_key)
        try:
            file_id = re.search(r'^.*/(.*)\.h5$', path, flags=re.MULTILINE).group(1)
            result = dynamodb.get_item(TableName=table, Key={"Id": {"N": file_id}})
        except ClientError as e:
            print(e.response["Error"]["Message"])
        else:
            name = file_id + ".h5"
            date = s3_obj_metadata["LastModified"].isoformat()
            size = str(s3_obj_metadata["ContentLength"])
            if "Item" in result:
                # Item exist in dynamo --> update the item with metadata and tags
                try:
                    attribute_updates = {"Name": {"Value": {"S": name}, "Action": "PUT"},
                                         "Path": {"Value": {"S": path}, "Action": "PUT"},
                                         "Date": {"Value": {"S": date}, "Action": "PUT"},
                                         "Size": {"Value": {"N": size}, "Action": "PUT"}}
                    dynamodb.update_item(TableName=table, Key={"Id": {"N": file_id}}, AttributeUpdates=attribute_updates)
                except ClientError as e:
                    print(e.response["Error"]["Message"])
                else:
                    print("The item has been updated")
            else:
                # Item does not exist in dynamo --> create the item with metadata and tags
                item_dict = {"Id": {"S": file_id}, "Path": {"S": path}, "Name": {"S": name}, "Date": {"S": date},
                             "Size": {"N": size}}
                try:
                    dynamodb.put_item(TableName=table, Item=item_dict)
                except ClientError as e:
                    print(e.response["Error"]["Message"])
                else:
                    print("The item has been added")


if __name__ == "__main__":
    main()
