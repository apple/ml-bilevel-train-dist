#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Helper tool to extract text from rcv1 xml."""

import sys
import xml.etree.ElementTree as ET

args = sys.argv[1:]
for xml_file in args:
    try:
        with open(xml_file, "r") as fin:
            xml_data = fin.read()
            # remove strings breaking the xml parser.
            for r in (("\n", ""), ("<p>", ""), ("</p>", " "), ("\t", " ")):
                xml_data = xml_data.replace(*r)
        root = ET.fromstring(xml_data)
        for child in root:
            if child.tag == "text":
                print(child.text)
                break
    # skip all parsing errors
    except Exception:
        pass
