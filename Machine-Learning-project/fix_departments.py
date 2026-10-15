import sqlite3

DB = r'C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\civicfix.db'

DEPT_MAP = {
    'Illegal Parking'        : 'NYPD - Traffic Division',
    'Noise - Residential'    : 'NYPD - Community Affairs',
    'Noise - Street/Sidewalk': 'NYPD - Community Affairs',
    'Noise - Vehicle'        : 'NYPD - Traffic Division',
    'Blocked Driveway'       : 'NYPD - Traffic Division',
    'Derelict Vehicle'       : 'NYPD - Traffic Division',
    'Graffiti'               : 'NYPD - Community Affairs',
    'UNSANITARY CONDITION'   : 'Dept. of Sanitation (DSNY)',
    'Dirty Condition'        : 'Dept. of Sanitation (DSNY)',
    'Street Condition'       : 'Dept. of Transportation (DOT)',
    'Street Light Condition' : 'Dept. of Transportation (DOT)',
    'Pothole'                : 'Dept. of Transportation (DOT)',
    'Sidewalk Condition'     : 'Dept. of Transportation (DOT)',
    'Water Maintenance'      : 'Dept. of Environmental Protection (DEP)',
    'HEAT/HOT WATER'         : 'Dept. of Housing Preservation (HPD)',
    'PLUMBING'               : 'Dept. of Housing Preservation (HPD)',
    'PAINT/PLASTER'          : 'Dept. of Housing Preservation (HPD)',
    'Other'                  : 'Municipal Services',
}

conn = sqlite3.connect(DB)

rows = conn.execute(
    "SELECT ticket_id, complaint_type FROM complaints WHERE department IS NULL OR department = ''"
).fetchall()

print(f"Found {len(rows)} complaints with no department assigned.")

for ticket_id, complaint_type in rows:
    dept = DEPT_MAP.get(complaint_type, 'Municipal Services')
    conn.execute(
        "UPDATE complaints SET department=? WHERE ticket_id=?",
        (dept, ticket_id)
    )
    print(f"  {ticket_id}  ({complaint_type})  ->  {dept}")

conn.commit()
conn.close()
print("\nAll departments updated successfully.")
