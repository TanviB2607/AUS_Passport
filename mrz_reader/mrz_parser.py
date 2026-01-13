def parse_mrz(line1, line2):
    data = {}

    data["type"] = line1[0]
    data["code_of_issue"] = line1[2:5]

    name_section = line1[5:]
    surname, given = name_section.split("<<", 1)

    data["surname"] = surname.replace("<", " ").strip()
    data["name"] = given.replace("<", " ").strip()

    data["doc no"] = line2[0:9].replace("<", "")
    data["nationality"] = line2[10:13]
    data["DOB"] = line2[13:19]
    data["gender"] = line2[20]
    data["date_of_expiry"] = line2[21:27]

    return data
