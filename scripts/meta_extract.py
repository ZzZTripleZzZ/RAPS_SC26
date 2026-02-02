# usage: cat date=2023-09-07.out | python raps/utils/extract.py
import sys
import re

# Read all input
data = sys.stdin.read()

# Define patterns for telemetry metadata
#patterns = [
#    "num_jobs (\d+)",
#    "Average job arrival time is: ([\d.]+)s",
#    "Average wall time is: ([\d.]+)s",
#    "Nodes required \(avg\): ([\d.]+)",
#    "Nodes required \(max\): (\d+)",
#    "Nodes required \(std\): ([\d.]+)"
#]
#extracted_numbers = [re.search(pattern, data).group(1) for pattern in patterns]
#print(", ".join(extracted_numbers))

# Define patterns for simulation output
patterns = [
    "\"num_samples\": (\d+)",
    "\"jobs completed\": (\d+)",
    "\"throughput\": \"([\d.]+) jobs/hour",
    "\"average power\": \"([\d.]+) MW",
    "\"min loss\": \"([\d.]+) MW \(([\d.]+)%\)",
    "\"average loss\": \"([\d.]+) MW \(([\d.]+)%\)",
    "\"max loss\": \"([\d.]+) MW \(([\d.]+)%\)",
    "\"system power efficiency\": \"([\d.]+)",
    "\"total energy consumed\": \"([\d.]+) MW",
    "\"carbon emissions\": \"([\d.]+) metric tons CO2",
    "\"total cost\": \"\$(\d+\.\d+)"
]

# Extract and print the numbers in the desired format
extracted_numbers = []
for pattern in patterns:
    matches = re.findall(pattern, data)
    for match in matches:
        # This will flatten the tuple if the match includes both values and percentages
        extracted_numbers.extend(match if isinstance(match, tuple) else [match])

print(", ".join(extracted_numbers))

