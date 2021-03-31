# sustainable_building_project_analysis

## Use this command to locate non utf-8 characters in JSON files:
    grep -axv '.*' file.txt

## Use verify_json.txt to recursively check all JSON files:
    python3 verify_json.py > invalid_json.txt

    Notes:
        -Output is piped to invalid_json.txt alphabetically
        -All error messages must be resolved to use analysis.py

## Use analysis.py to perform linear regression on data files:
    python3 analysis.py

## File Structure:
    -Data directory must contain subdirectories with valid json files
    -All other files and tools in main directory