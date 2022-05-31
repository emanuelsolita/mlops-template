#!/usr/bin/env python3

def init():
    print("Service initialized")

def run(raw_data):
    print(f"Got this data: {raw_data}")
    return {"id":1}#raw_data
