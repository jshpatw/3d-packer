import os
from diagnose_overlaps import check_overlap_robust

if __name__ == "__main__":
    dims = (5.898, 2.352, 2.393)
    
    files_to_check = [
        "sage_tetrapod/export_results/04_Frontier_Research/SAGE_Engine_23Unit.glb",
        "sage_tetrapod/export_results/04_Frontier_Research/Frontier_24Unit_Result.glb"
    ]
    
    for path in files_to_check:
        print(f"\n--- AUDITING: {path} ---")
        if os.path.exists(path):
            check_overlap_robust(path, dims)
        else:
            print(f"File {path} not found.")
