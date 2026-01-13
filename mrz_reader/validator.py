def normalize(text):
    if not text:
        return ""
    return text.lower().replace(" ", "").replace("<", "")

def cross_validate(mrz_data, ocr_data):

    checks = {}

    checks["type_match"] = normalize(mrz_data.get("type")) == normalize(ocr_data.get("type"))
    checks["name_match"] = normalize(mrz_data.get("name")) in normalize(ocr_data.get("name"))
    checks["surname_match"] = normalize(mrz_data.get("surname")) in normalize(ocr_data.get("surname"))

    checks["doc_no_match"] = normalize(mrz_data.get("doc no")) in normalize(ocr_data.get("doc no"))
    checks["nationality_match"] = normalize(mrz_data.get("nationality")) == normalize(ocr_data.get("nationality"))

    checks["dob_match"] = mrz_data.get("DOB") in normalize(ocr_data.get("DOB"))
    checks["gender_match"] = mrz_data.get("gender") == normalize(ocr_data.get("gender"))

    checks["expiry_match"] = mrz_data.get("date_of_expiry") in normalize(ocr_data.get("date_of_expiry"))
    checks["issue_country_match"] = normalize(mrz_data.get("code_of_issue")) == normalize(ocr_data.get("code_of_issue"))

    return checks
