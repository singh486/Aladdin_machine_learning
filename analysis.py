import json
import glob
import os
import pandas as pd


def concat_files():
    read_files = glob.glob("FD_5/*.json")
    with open("merged_file.json", "wb") as outfile:
        outfile.write('[{}]'.format(
            ','.join([open(f, "r").read() for f in read_files])).encode())


def parse_dictionary():
    # Set path
    os.chdir(
        "FD_5")

    # Dictionary to store action counts

    cDictionary = {'Add Door': 0, 'Remove Door': 0, 'Edit Door': 0,
                   'Add Floor': 0, 'Remove Floor': 0, 'Edit Floor': 0,
                   'Add Foundation': 0, 'Remove Foundation': 0, 'Edit Foundation': 0,
                   'Add Wall': 0, 'Remove Wall': 0, 'Edit Wall': 0, 'Type Change of Wall': 0,
                   'Change Thickness for Selected Wall': 0, 'Change Thickness for Walls on Selected Foundation': 0,
                   'Change Thickness for All Walls': 0, 'Change Height for Selected Wall': 0,
                   'Change Height for Walls on Selected Foundation': 0, 'Change Height for All Walls': 0,
                   'Change Height for Connected Walls': 0,
                   'Add Window': 0, 'Remove Window': 0, 'Edit Window': 0, 'Paste Window': 0,
                   'Set Size for Selected Window': 0,
                   'Add CustomRoof': 0, 'Add HipRoof': 0, 'Add PyramidRoof': 0, 'Add ShedRoof': 0,
                   'Add GambrelRoof': 0, 'Remove CustomRoof': 0, 'Remove HipRoof': 0, 'Remove PyramidRoof': 0,
                   'Remove ShedRoof': 0, 'Remove GambrelRoof': 0, 'Edit CustomRoof': 0, 'Edit HipRoof': 0,
                   'Edit PyramidRoof': 0, 'Edit ShedRoof': 0, 'Edit GambrelRoof': 0,
                   'Add SolarPanel': 0, 'Remove SolarPanel': 0, 'Edit SolarPanel': 0, 'Paste SolarPanel': 0,
                   'Rotate Solar Panel': 0, 'Choose Size for Selected Solar Panel': 0, 'Add SolarPanel Array': 0,
                   'Solar Cell Efficiency Change for Selected Solar Panel': 0,
                   'Solar Cell Efficiency Change for All Solar Panels on Selected Foundation': 0,
                   'Solar Cell Efficiency Change for All Solar Panels': 0,
                   'Inverter Efficiency Change for Selected Solar Panel': 0,
                   'Inverter Efficiency Change for All Solar Panels on Selected Foundation': 0,
                   'Inverter Efficiency Change for All Solar Panels': 0,
                   'Add Tree': 0, 'Remove Tree': 0, 'Move Tree': 0, 'Paste Tree': 0,
                   'Move Building': 0, 'Resize Building': 0, 'Rotate Building': 0, 'Remove Building': 0,
                   'Rescale Building': 0,
                   'Show Shadow': 0, 'Show Heliodon': 0, 'Show Heat Flux Vectors': 0, 'Animate Sun': 0,
                   'Graph Tab': 0, 'DailyEnvironmentalTemperature': 0, 'AnnualEnvironmentalTemperature': 0,
                   'Solar Potential': 0, 'Cost': 0, 'EnergyDailyAnalysis': 0, 'DailyEnergyGraph': 0,
                   'SolarDailyAnalysis': 0, 'SolarAnnualAnalysis': 0, 'GroupDailyAnalysis': 0,
                   'GroupAnnualAnalysis': 0,
                   'Change City': 0, 'Change Latitude': 0, 'Change Date': 0, 'Change Time': 0,
                   'Adjust Thermostat': 0, 'U-Factor Change for Selected Part': 0,
                   'U-Factor Change for Whole Building': 0,
                   'Color Change for Selected Part': 0, 'Color Change for Whole Building': 0
                   }

    for file in glob.glob("*.json"):
        with open(file) as f:

            try:
                data = json.load(f)

                # For each JSON file count particular activities
                for item in data["Activities"]:
                    for subitem in item:
                        if subitem in cDictionary:
                            cDictionary[subitem] = cDictionary[subitem] + 1

            except Exception as e:
                print("Error with ", file)
                print(str(e))

    # New merged dictionary

    mDictionary = {'Door': cDictionary['Add Door'] + cDictionary['Remove Door'] + cDictionary['Edit Door'],
                   'Floor': cDictionary['Add Floor'] + cDictionary['Remove Floor'] + cDictionary['Edit Floor'],
                   'Foundation': cDictionary['Add Foundation'] + cDictionary['Remove Foundation'] +
                   cDictionary['Edit Foundation'],
                   'Wall': cDictionary['Add Wall'] + cDictionary['Remove Wall'] + cDictionary['Edit Wall'] +
                   cDictionary['Type Change of Wall'] + cDictionary['Change Thickness for Selected Wall'] +
                   cDictionary['Change Thickness for Walls on Selected Foundation'] +
                   cDictionary['Change Thickness for All Walls'] + cDictionary['Change Height for Selected Wall'] +
                   cDictionary['Change Height for Walls on Selected Foundation'] +
                   cDictionary['Change Height for All Walls'] +
                   cDictionary['Change Height for Connected Walls'],
                   'Window': cDictionary['Add Window'] + cDictionary['Remove Window'] + cDictionary['Edit Window'] +
                   cDictionary['Paste Window'] +
                   cDictionary['Set Size for Selected Window'],
                   'Roof': cDictionary['Add CustomRoof'] + cDictionary['Add HipRoof'] + cDictionary['Add PyramidRoof'] +
                   cDictionary['Add ShedRoof'] + cDictionary['Add GambrelRoof'] + cDictionary['Remove CustomRoof'] +
                   cDictionary['Remove HipRoof'] + cDictionary['Remove PyramidRoof'] + cDictionary['Remove ShedRoof'] +
                   cDictionary['Remove GambrelRoof'] + cDictionary['Edit CustomRoof'] + cDictionary['Edit HipRoof'] +
                   cDictionary['Edit PyramidRoof'] +
                   cDictionary['Edit ShedRoof'] +
                   cDictionary['Edit GambrelRoof'],
                   'Solar Panel': cDictionary['Add SolarPanel'] + cDictionary['Remove SolarPanel'] +
                   cDictionary['Edit SolarPanel'] + cDictionary['Paste SolarPanel'] + cDictionary['Rotate Solar Panel'] +
                   cDictionary['Choose Size for Selected Solar Panel'] + cDictionary['Add SolarPanel Array'] +
                   cDictionary['Solar Cell Efficiency Change for Selected Solar Panel'] +
                   cDictionary['Solar Cell Efficiency Change for All Solar Panels on Selected Foundation'] +
                   cDictionary['Solar Cell Efficiency Change for All Solar Panels'] +
                   cDictionary['Inverter Efficiency Change for Selected Solar Panel'] +
                   cDictionary['Inverter Efficiency Change for All Solar Panels on Selected Foundation'] +
                   cDictionary['Inverter Efficiency Change for All Solar Panels'],
                   'Tree': cDictionary['Add Tree'] + cDictionary['Remove Tree'] + cDictionary['Move Tree'] +
                   cDictionary['Paste Tree'],
                   'Building': cDictionary['Move Building'] + cDictionary['Resize Building'] +
                   cDictionary['Rotate Building'] +
                   cDictionary['Remove Building'] +
                   cDictionary['Rescale Building'],
                   'Analysis': cDictionary['Show Shadow'] + cDictionary['Show Heliodon'] +
                   cDictionary['Show Heat Flux Vectors'] + cDictionary['Animate Sun'] + cDictionary['Graph Tab'] +
                   cDictionary['DailyEnvironmentalTemperature'] + cDictionary['AnnualEnvironmentalTemperature'] +
                   cDictionary['Solar Potential'] + cDictionary['Cost'] + cDictionary['EnergyDailyAnalysis'] +
                   cDictionary['DailyEnergyGraph'] + cDictionary['SolarDailyAnalysis'] +
                   cDictionary['SolarAnnualAnalysis'] + cDictionary['GroupDailyAnalysis'] +
                   cDictionary['GroupAnnualAnalysis'],
                   'Parameters': cDictionary['Change City'] + cDictionary['Change Latitude'] +
                   cDictionary['Change Date'] + cDictionary['Change Time'],
                   'Thermal': cDictionary['Adjust Thermostat'] + cDictionary['U-Factor Change for Selected Part'] +
                   cDictionary['U-Factor Change for Whole Building'],
                   'Color': cDictionary['Color Change for Selected Part'] +
                   cDictionary['Color Change for Whole Building']
                   }
    df = pd.DataFrame(list(mDictionary.items()),columns = ['Actions','Counts']) 
    print(df)


if __name__ == "__main__":
    # concat_files()
    parse_dictionary()
    # df = pd.read_json(r'merged_file.json')
    # print(df)
