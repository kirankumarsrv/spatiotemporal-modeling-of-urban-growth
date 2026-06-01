from graphviz import Digraph

dot = Digraph(
    comment='Normalization Diagram',
    format='png',
    graph_attr={
        'rankdir': 'TB',
        'fontname': 'Times New Roman'
    },
    node_attr={
        'shape': 'record',
        'fontname': 'Times New Roman'
    }
)

# ---------- TABLE DEFINITIONS ----------

dot.node(
    'PERSON',
    '{PERSON | Person_ID (PK) | Name | Rank | Age | Gender | Service_Years}'
)

dot.node(
    'THERAPIST',
    '{THERAPIST | Therapist_ID (PK) | Name | Qualification | Specialization | Experience}'
)

dot.node(
    'SCENARIO',
    '{SCENARIO | Scenario_ID (PK) | Scenario_Type | Environment | Assigned_Date}'
)

dot.node(
    'REACTION',
    '{REACTION | Reaction_ID (PK) | Reaction_Type | Physical_Response}'
)

dot.node(
    'REPORT',
    '{REPORT | Report_ID (PK) | Reaction_ID (FK) | Person_ID (FK) | Therapist_ID (FK) | '
    'Avoidance | Re_Experiencing | Negative_Alterations | Hyperarousal}'
)

# ---------- FUNCTIONAL DEPENDENCIES ----------

dot.edge('PERSON', 'PERSON', label='Person_ID → All Attributes')
dot.edge('THERAPIST', 'THERAPIST', label='Therapist_ID → All Attributes')
dot.edge('SCENARIO', 'SCENARIO', label='Scenario_ID → All Attributes')
dot.edge('REACTION', 'REACTION', label='Reaction_ID → All Attributes')
dot.edge('REPORT', 'REPORT', label='Report_ID → All Attributes')

# ---------- NORMAL FORM SUMMARY ----------

dot.node(
    'NF',
    '{NORMAL FORMS | '
    '1NF : ✔ | '
    '2NF : ✔ | '
    '3NF : ✔ | '
    'BCNF : ✔}'
)

dot.edge('REPORT', 'NF')

# ---------- RENDER ----------

# Save the DOT source code without rendering (doesn't require Graphviz installation)
with open('Normalization_Person_Therapist_Project.dot', 'w', encoding='utf-8') as f:
    f.write(dot.source)

print("Normalization diagram source code saved to .dot file.")
print("To generate the image, install Graphviz from: https://graphviz.org/download/")
print("Or use an online viewer like: https://dreampuf.github.io/GraphvizOnline/")
