import json

with open('PredictionPipeline/Prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('pose_engine_code.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        source = ''.join(cell.get('source', []))
        if 'class PoseRetrievalEngine' in source:
            out.write(f"=== Cell {i} ===\n")
            out.write(source)
            out.write("\n" + "="*80 + "\n")
    
print("Extracted PoseRetrievalEngine code to pose_engine_code.txt")

