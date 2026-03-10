import torch
import argparse
import json
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def predict(text, model_path="mstyle-detector-final"):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Process text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    
    label = "AI Generated" if predicted_class_id == 1 else "Human Written"
    
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence = probs[0][predicted_class_id].item() * 100
    
    print(f"\nText: \"{text[:100]}...\"")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%\n")

#if __name__ == "__main__":
   # parser = argparse.ArgumentParser()
    #parser.add_argument("--model_path", type=str, default="./models/mstyle-detector-final", help="Path to fine-tuned model")
    #parser.add_argument("--text", type=str, required=True, help="Text to classify")
    
    #args = parser.parse_args()
    #predict(args.text, args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Cambiamos para que sea compatible con el sistema del concurso
    parser.add_argument("-i", "--input", type=str, help="Ruta al archivo JSONL de entrada")
    parser.add_argument("-o", "--output", type=str, help="Carpeta para guardar predictions.jsonl")
    parser.add_argument("--model_path", type=str, default="./models/mstyle-detector-final")
    
    args = parser.parse_args()
    
    # Si el concurso envía -i, ejecutamos lógica de lote, si no, lógica de texto único
    if args.input:
        # 1. Cargar modelo una sola vez para ser eficientes
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print("Iniciando carga de modelo...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        results = []
        # 2. Leer archivo JSONL
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Procesar texto
                inputs = tokenizer(item['text'], return_tensors="pt", truncation=True, max_length=512)
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                score = probs[0][1].item() # Probabilidad de ser IA
                
                results.append({"id": item['id'], "score": score})

        # 3. Guardar resultados
        with open(os.path.join(args.output, 'predictions.jsonl'), 'w') as out:
            for res in results:
                out.write(json.dumps(res) + '\n')
        print("¡Proceso completado exitosamente!")
    else:
        # Tu lógica actual de una sola predicción
        predict("Texto de prueba", args.model_path)