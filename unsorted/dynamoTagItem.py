import argparse
import re
import boto3
from botocore.errorfactory import ClientError
import ast


def main():
    # ------------------------------------- User-friendly command-line interfaces --------------------------------------
    parser = argparse.ArgumentParser(description="Add or modify a tag of an object sync between S3 and dynamo")
    parser.add_argument("path", type=str, action="store", help="The path of the object within the table")
    parser.add_argument("tag", type=str, action="store", help="The tag attribute to add or modify")
    parser.add_argument("value", type=str, action="store", help="The new value of the tag")
    parser.add_argument("type", type=str, action="store",
                        choices=["S", "N", "B", "SS", "NS", "BS", "M", "L", "NULL", "BOOL"],
                        help="The type of the new value of the tag : {string, number, binary, string set, number set,"
                             "binary set, dict, list, null, bool}")

    args = parser.parse_args()

    # Check the value and the type are concordant
    if args.type in ["SS", "BS"]:
        check = re.search(r'^\[(?:.+,)*(?:.+)+\]$', args.value, flags=re.MULTILINE)
        if not check:
            print("The value doesn't match the type. Expected [\"string1\", \"string2\", ...]")
            exit()
    elif args.type == "NS":
        check = re.search(r'^\[\s*(?:(-?\d*\.?\d*)\s*,\s*(-?\d*\.?\d*))+\s*\]$', args.value, flags=re.MULTILINE)
        if not check:
            print("The value doesn't match the type. Expected [number1, number2, ...]")
            exit()
    elif args.type in ["NULL", "BOOL"]:
        if args.value not in ["true", "false"]:
            print("The value doesn't match the type. Expected true or false")
            exit()
    else:
        pass

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
    path = args.path
    tag = args.tag
    value = args.value
    tag_type = args.type
    table = "S3Sync"
    dynamodb = boto3.client("dynamodb", region_name=region, aws_access_key_id=public_key,
                            aws_secret_access_key=private_key)
    try:
        result = dynamodb.get_item(TableName=table, Key={"Path": {"S": path}})
    except ClientError as e:
        print(e.response["Error"]["Message"])
    else:
        if "Item" in result:
            try:
                if tag_type in ["SS", "NS", "BS"]:
                    value = ast.literal_eval(value)
                dict_v = {args.type: value}
                attribute_updates = {tag: {"Value": dict_v, "Action": "PUT"}}
                dynamodb.update_item(TableName=table, Key={"Path": {"S": path}}, AttributeUpdates=attribute_updates)
            except ClientError as e:
                print(e.response["Error"]["Message"])
            else:
                print("The item has been updated")
        else:
            print("Item not found")


if __name__ == "__main__":
    main()
