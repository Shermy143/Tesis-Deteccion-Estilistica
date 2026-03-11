import argparse
import json
import os
import glob

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="script")
    parser.add_argument("-i", "--input", required=True, help="Ruta al dataset de entrada (directorio).")
    parser.add_argument("-o", "--output", required=True, help="Ruta donde se guardará el archivo de salida.")
    return parser.parse_args()

def run_inference(input_dir, output_dir):
    # 1. Asegurarnos de que el directorio de salida existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "predictions.jsonl")
    
    # 2. Buscar archivos de entrada (TIRA suele montar el dataset como archivos .jsonl)
    input_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    
    if not input_files:
        print(f"No se encontraron archivos .jsonl en {input_dir}. Generando salida dummy para validación.")
        # Generar una línea dummy para que TIRA no falle el 'smoke test'
        with open(output_file, "w") as f:
            dummy_result = {"id": "dummy-test-001", "label": "human"}
            f.write(json.dumps(dummy_result) + "\n")
        return

    # 3. Procesar los archivos reales
    with open(output_file, "w") as out_f:
        for file_path in input_files:
            with open(file_path, "r") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    
                    data = json.loads(line)
                    doc_id = data.get("id", "unknown")
                    
                    # --- AQUÍ VA TU LÓGICA DE MODELO ---
                    # Por ahora, una predicción fija para que el formato sea válido
                    prediction = "human" 
                    # ----------------------------------
                    
                    # Escribir en formato JSONL
                    result = {"id": doc_id, "label": prediction}
                    out_f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    args = parse_args()

    print(f"Leyendo datos desde: {args.input}")
    print(f"Escribiendo resultados en: {args.output}")

    try:
        run_inference(args.input, args.output)
        print(f"Proceso completado con éxito. Archivo generado en {args.output}/predictions.jsonl")
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        exit(1)
